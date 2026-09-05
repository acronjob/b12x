# GLM-5.3 Flash R26 serving patches

**This checkout is the experimental branch. These notes describe its serving
base; see [experimental scope and limits](glm53-r26-experiments.md).**

This branch preserves the source-locked R26 integration and the local patches
used in the validated runtime. Use the matching `patches/r26-serving-20260905`
branch in both `acronjob/vllm` and `acronjob/b12x`.

Runtime image:

```text
voipmonitor/vllm:jovian-judgement-community-20260905-r26@sha256:d0592ea9d73cac5aadb151a58bbb43cf7aff03829d46bb4f4ba7396aaef67c68
```

vLLM includes upstream `dev/jovian-judgement` commit
`3512b066e7796128c0c380ccc558182960f2f0ea`, retaining R26's integration features.
The compatibility guard skips the legacy BF16-only draft-head copy when the
effective head is owned or runtime-quantized. Both legacy and V2 callers are
covered by targeted tests.

## Local changes

- vLLM shards token-local mHC work for eligible target forwards of at least
  4096 tokens. Tensor-parallel attention and expert placement stay unchanged.
- vLLM fuses sparse MLA index mapping, causal counts and tail masking for
  eligible selections of at most 512 entries. GLM's 2051-wide selection keeps
  its fallback; this bounded helper receives no GLM serving speedup credit.
- b12x uses posted writes for the eligible TP4 plain all-reduce path, preserving
  each destination's FP32 summation order and final BF16/FP16 conversion.

The two opt-in settings used by the measured runtime are:

```bash
export VLLM_GLM53_MHC_TOKEN_SHARD=1
export B12X_PCIE_PLAIN_TP4_REMOTE_PUSH=1
```

Keep the new upstream recipe flags unset to preserve automatic eligibility
fallback. Do not turn every new default into an explicit environment override.
The selected profile uses TP4/DCP4, DFlash2 K7, FP8 MLA KV, batch 8192, memory
utilization 0.93 and FULL_DECODE_ONLY. Use four GPUs on the same physical switch.
The tests did not alter power, clocks, memory clocks or fan settings.

## Evidence and limits

The complete operation was tested through source-file overlays on the exact
image above; no vLLM rebuild was performed. The per-repository manifest lists
byte hashes of those deployed sources. A new source-tree build has not been
qualified by these measurements.

Matched old-patched / updated-only / updated-patched / old-patched runs used
identical launch settings, prompt token IDs and cache invariants, with two
warmups and seven measured repetitions per case. Existing research retained
7.47–8.55% faster client-prefill throughput versus updated upstream alone.
Updated patched prefill remained within 0.4% of both old patched runs.
User-template 8196/32317-token prompts measured 10745.55/11827.48 tokens/s.
C1 generation was 267.21 tokens/s; C4 aggregate was 477.23 tokens/s. C1 moved
-1.39% to -5.74% and C4 +2.33% to +4.50% against the repeated baselines; the
unchanged C1 baseline itself shifted 4.40%. Speculative acceptance varies, so
these data establish no general decode-throughput gain from the upstream update.

Updated patched KV capacity matched the repeated baseline at 14270006 tokens.
Sampled whole-process memory maxima rose 140–160 MiB per GPU versus that
baseline, including startup, graph capture and reservations. Sampling can miss
brief peaks and is not an instantaneous allocation bound.

The upstream target head intentionally uses MXFP8 weights with BF16 activations,
shared by DFlash2. General model-quality equivalence is unproven. Five serving
smokes and an uncached 32768-token deployment smoke passed. A fixed supplemental
JSON-answer-suffix diagnostic scored 5/6 for the updated patched arm because one
answer added a Markdown fence, versus 6/6 elsewhere; strict full-content JSON
was 0/6 in every arm. Those failures were retained, not rescored.

The exported sources passed a fresh CPU run of 139 tests, including both
merged model-test sets and draft-head coexistence tests. Two GPU cases were
deselected in that CPU-only run. Earlier targeted component tests passed
135 CPU and 19 GPU cases. One optional
NVFP4 `VLLM_LM_HEAD_A16=0` numerical test failed its unchanged tolerance; that
mode is disabled in the measured profile and remains unqualified. Default A16
cases passed. Runtime imports and deployed file hashes matched all 14 overlays.

The branch exports source, portable tests and this result summary. Local machine
inventories, request/response logs, profiler captures and host-specific launch
files are not part of the export. Unfinished metadata and L2 experiments belong
on the separate experimental branch and are not enabled here.
