"""Pooled-index selection transforms for paged sparse MLA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _expand_pooled_topk_to_physical_slots_kernel(
    pool_indices,
    last_token_positions,
    request_ids,
    block_table,
    output,
    active_counts,
    pool_stride,
    block_table_stride,
    output_stride,
    max_num_blocks,
    num_cache_blocks,
    HISTORY_TOKENS: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_STRIDE_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    column = tile * BLOCK_COLS + tl.arange(0, BLOCK_COLS)
    mask = column < OUTPUT_WIDTH

    sequence_length = tl.load(last_token_positions + row).to(tl.int64) + 1
    complete_pools = sequence_length // POOL_SIZE
    selected_pools = tl.minimum(complete_pools, HISTORY_TOKENS // POOL_SIZE)
    selected_history_tokens = selected_pools * POOL_SIZE
    tail_start = complete_pools * POOL_SIZE
    history = column < selected_history_tokens
    pool_column = column // POOL_SIZE
    pool_offset = column % POOL_SIZE
    pool_id = tl.load(
        pool_indices + row * pool_stride + pool_column,
        mask=mask & history,
        other=-1,
    ).to(tl.int64)
    history_value = tl.where(pool_id >= 0, pool_id * POOL_SIZE + pool_offset, -1)
    tail_offset = column - selected_history_tokens
    tail_count = sequence_length - tail_start
    in_tail = (tail_offset >= 0) & (tail_offset < tail_count)
    logical_token = tl.where(
        history,
        history_value,
        tl.where(in_tail, tail_start + tail_offset, -1),
    )

    request = tl.load(request_ids + row).to(tl.int64)
    block_id = logical_token // BLOCK_SIZE
    in_block = logical_token - block_id * BLOCK_SIZE
    valid = (logical_token >= 0) & (block_id < max_num_blocks)
    page = tl.load(
        block_table + request * block_table_stride + block_id,
        mask=mask & valid,
        other=-1,
    ).to(tl.int64)
    valid &= (page >= 0) & (page < num_cache_blocks)
    physical_token = tl.where(
        valid,
        page * BLOCK_STRIDE_ROWS + in_block,
        -1,
    ).to(tl.int32)
    tl.store(output + row * output_stride + column, physical_token, mask=mask)

    active_count = selected_pools * POOL_SIZE + tail_count
    tl.store(active_counts + row, active_count, mask=tile == 0)


def expand_pooled_topk_to_physical_slots(
    pool_indices: torch.Tensor,
    last_token_positions: torch.Tensor,
    request_ids: torch.Tensor,
    block_table: torch.Tensor,
    output: torch.Tensor,
    active_counts: torch.Tensor,
    *,
    pool_size: int,
    block_size: int,
    block_stride_rows: int,
    num_cache_blocks: int,
) -> None:
    """Expand selected pools and the live tail into physical cache slots.

    The caller supplies request-relative pool IDs, scheduler metadata, and
    caller-owned outputs. Selected pools and the live tail occupy one contiguous
    prefix of the output. The output width must cover every selected pool plus
    at most ``pool_size - 1`` unpooled tail tokens. Physical-slot arithmetic is
    performed in 64 bits and the result remains an int32 sparse-MLA index.
    """
    tensors = (
        pool_indices,
        request_ids,
        block_table,
        output,
        active_counts,
    )
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("pooled sparse-selection metadata must use int32 tensors")
    if last_token_positions.dtype != torch.int64:
        raise TypeError("last_token_positions must use int64")
    if pool_indices.ndim != 2 or int(pool_indices.shape[1]) < 1:
        raise ValueError("pool_indices must be a non-empty rank-two tensor")
    rows, pool_topk = map(int, pool_indices.shape)
    if last_token_positions.shape != (rows,) or request_ids.shape != (rows,):
        raise ValueError("pooled selection metadata must have one entry per row")
    if block_table.ndim != 2 or min(map(int, block_table.shape)) < 1:
        raise ValueError("block_table must be a non-empty rank-two tensor")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    output_width = pool_topk * pool_size + pool_size - 1
    if output.shape != (rows, output_width) or active_counts.shape != (rows,):
        raise ValueError(
            "pooled physical-selection outputs must have shapes "
            f"({rows}, {output_width}) and ({rows},)"
        )
    if block_size < 1 or block_stride_rows < block_size:
        raise ValueError("physical cache block geometry is invalid")
    max_physical_slot = (
        (int(num_cache_blocks) - 1) * int(block_stride_rows) + int(block_size) - 1
    )
    if num_cache_blocks < 1 or max_physical_slot > torch.iinfo(torch.int32).max:
        raise ValueError("physical cache slots exceed the int32 index range")
    all_tensors = (*tensors, last_token_positions)
    if any(tensor.device != pool_indices.device for tensor in all_tensors):
        raise ValueError("pooled sparse-selection tensors must share one device")
    if not pool_indices.is_contiguous() or not output.is_contiguous():
        raise ValueError("pool_indices and output must be contiguous")

    if rows:
        block_cols = 128
        _expand_pooled_topk_to_physical_slots_kernel[
            (rows, triton.cdiv(output_width, block_cols))
        ](
            pool_indices,
            last_token_positions,
            request_ids,
            block_table,
            output,
            active_counts,
            int(pool_indices.stride(0)),
            int(block_table.stride(0)),
            int(output.stride(0)),
            int(block_table.shape[1]),
            int(num_cache_blocks),
            HISTORY_TOKENS=pool_topk * pool_size,
            OUTPUT_WIDTH=output_width,
            POOL_SIZE=pool_size,
            BLOCK_SIZE=block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            BLOCK_COLS=block_cols,
            num_warps=4,
        )


__all__ = ["expand_pooled_topk_to_physical_slots"]


# Default-off R26 DCP selection experiment.
@triton.jit
def _expand_dcp_pools(
    pools, positions, requests, table, output, counts,
    pool_stride, table_stride0, table_stride1, output_stride,
    max_blocks, num_cache_blocks,
    DCP_SIZE: tl.constexpr, DCP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr, PAGE_SIZE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    p = tl.arange(0, 512)
    pool = tl.load(pools + row * pool_stride + p).to(tl.int64)
    token = pool * 4
    local = (token // (DCP_SIZE * INTERLEAVE)) * INTERLEAVE + token % INTERLEAVE
    owner = (token // INTERLEAVE) % DCP_SIZE
    block = local // PAGE_SIZE
    request = tl.load(requests + row).to(tl.int64)
    valid = (pool >= 0) & (owner == DCP_RANK) & (block >= 0) & (block < max_blocks)
    page = tl.load(table + request * table_stride0 + block * table_stride1,
                   mask=valid, other=-1).to(tl.int64)
    valid &= (page >= 0) & (page < num_cache_blocks)
    physical = page * PAGE_STRIDE + local % PAGE_SIZE
    flags = valid.to(tl.int32)
    offsets = (tl.cumsum(flags) - flags) * 4
    history_count = tl.sum(flags) * 4

    seq = tl.load(positions + row).to(tl.int64) + 1
    tail_start = (seq // 4) * 4
    tail_local = (tail_start // (DCP_SIZE * INTERLEAVE)) * INTERLEAVE + tail_start % INTERLEAVE
    tail_block = tail_local // PAGE_SIZE
    tail_valid = (seq > 0) & ((tail_start // INTERLEAVE) % DCP_SIZE == DCP_RANK)
    tail_valid &= (tail_block >= 0) & (tail_block < max_blocks)
    tail_page = tl.load(table + request * table_stride0 + tail_block * table_stride1,
                        mask=tail_valid, other=-1).to(tl.int64)
    tail_valid &= (tail_page >= 0) & (tail_page < num_cache_blocks)
    tail_count = tl.where(tail_valid, seq % 4, 0).to(tl.int32)
    active = history_count + tail_count

    # The invalid suffix and compact valid prefix have disjoint destinations.
    columns = tl.arange(0, 4096)
    tl.store(output + row * output_stride + columns, -1,
             mask=(columns < 2051) & (columns >= active))
    inner = tl.arange(0, 4)
    tl.store(output + row * output_stride + offsets[:, None] + inner[None, :],
             (physical[:, None] + inner[None, :]).to(tl.int32),
             mask=valid[:, None])
    tl.store(output + row * output_stride + history_count + inner,
             (tail_page * PAGE_STRIDE + tail_local % PAGE_SIZE + inner).to(tl.int32),
             mask=inner < tail_count)
    tl.store(counts + row, active)


def expand_dcp_pools(pools, positions, requests, table, output, counts, *,
                     dcp_size, dcp_rank, interleave, page_size,
                     page_stride, num_cache_blocks):
    """Write a compact stable prefix without allocating any tensor storage."""
    rows = pools.shape[0]
    if pools.shape != (rows, 512) or output.shape != (rows, 2051):
        raise ValueError("expected C4 selection shapes [rows,512] and [rows,2051]")
    if positions.shape != (rows,) or requests.shape != (rows,) or counts.shape != (rows,):
        raise ValueError("expected one metadata entry per row")
    if table.ndim != 2 or min(table.shape) < 1:
        raise ValueError("expected a nonempty request page table")
    if dcp_size not in (1, 2, 4) or not 0 <= dcp_rank < dcp_size:
        raise ValueError("unsupported DCP geometry")
    if page_size < 1 or interleave < 4 or interleave % 4 or page_size % interleave:
        raise ValueError("C4 ownership must align to pages and interleave groups")
    if page_stride < page_size or num_cache_blocks < 1:
        raise ValueError("invalid page geometry")
    if (num_cache_blocks - 1) * page_stride + page_size - 1 > 2**31 - 1:
        raise ValueError("physical slot IDs exceed the int32 ABI")
    metadata = (pools, requests, table, output, counts)
    if any(x.dtype != torch.int32 for x in metadata) or positions.dtype != torch.int64:
        raise TypeError("selection metadata must be int32; positions must be int64")
    if any(x.device != pools.device for x in (*metadata, positions)):
        raise ValueError("all tensors must share one device")
    if any(not x.is_contiguous() for x in (pools, positions, requests, output, counts)):
        raise ValueError("selection arrays must be contiguous")
    if rows:
        _expand_dcp_pools[(rows,)](
            pools, positions, requests, table, output, counts,
            pools.stride(0), table.stride(0), table.stride(1), output.stride(0),
            table.shape[1], num_cache_blocks,
            DCP_SIZE=dcp_size, DCP_RANK=dcp_rank, INTERLEAVE=interleave,
            PAGE_SIZE=page_size, PAGE_STRIDE=page_stride, num_warps=4,
        )
