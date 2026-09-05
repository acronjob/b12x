# R26 experimental metadata and prefetch patches

This branch builds on `patches/r26-serving-20260905`. Pair this branch in
`acronjob/vllm` and `acronjob/b12x`. The new gates all default off. These
experiments have component evidence but **no completed serving A/B/A
qualification**. Their component ratios are not model-throughput gains.

| Experiment | Repository/source | Optional gate | Evidence |
| --- | --- | --- | --- |
| Direct pooled DCP4 physical selection | b12x `pooled_selection.py`, vLLM `pooled_indexer.py` | `VLLM_GLM53_DCP_POOLED_PHYSICAL=1` | 252 exact component cases; direct decode mapping avoids an intermediate logical mapping |
| Width-2051 CKV mapping/count/mask | vLLM `b12x_mla_sparse.py` | `VLLM_GLM53_CKV_METADATA_2051=1` | Integer/producer cases and FP8 attention reference bounds pass; corrected causal-length initialization retained |
| Flat dense/router L2 hint traversal | vLLM `l2_prefetch.py` | `VLLM_GLM53_L2_PREFETCH_FLAT=1` | CPU chunk oracle and isolated GPU address/JIT/graph checks pass; real consumer and serving impact remain unqualified |

The four experimental source files match their existing component-qualified
snapshots byte for byte. Their original base files also match the current
serving branch; no textual rebase changes were needed. The combined
metadata-plus-L2 configuration has not been benchmarked.

Direct pooled mapping reuses the selected-index/count storage and preserves
pure-decode admission and logical fallback. The wide CKV helper reuses existing
buffers and applies only to eligible gathered-cache prefill; it is not the live
DCP4 decode kernel. Selected-token ordering can differ from the old atomic
compaction, so bitwise attention equivalence is not claimed. The L2 change
preserves hint address/count coverage and stream boundaries while adding small
immutable CTA descriptors; their whole-model allocation cost remains unmeasured.

Current host integration checks in vLLM can run without GPU imports:

```bash
.venv/bin/python validation/glm53_r26/check_pooled_routing.py
.venv/bin/python validation/glm53_r26/check_ckv_branches.py
.venv/bin/python validation/glm53_r26/check_l2_descriptors.py
```

These exercise actual host branches and the actual descriptor function with
independent scalar oracles. They do not establish device execution, attention
quality, available KV capacity or serving speed. The earlier failed CKV startup
was fixed before this source snapshot; its serving retest was interrupted
before timing. No deployment recommendation follows from this export.

`serving_manifest.json` records the serving base; overlay its entries with
`experimental_manifest.json` to describe the current branch's runtime files.
The serving notes remain a record of the serving branch, not this unqualified
experimental combination. No hardware settings are changed by these patches.
