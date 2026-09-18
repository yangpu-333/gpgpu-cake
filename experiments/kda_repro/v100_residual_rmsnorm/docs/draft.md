# Initial agent draft

## Baseline and validation

`src/reference.py` is the correctness oracle. `validate.py` checks the two
forward outputs against an independent expression and runs backward from a
scalar loss. `benchmark.py` records CUDA-event medians for representative
`[64, 768]`, `[64, 1024]`, and `[64, 4096]` FP16 cases.

## Risks and unknowns

- The V100 is sm70, whereas the public FlashInfer contest instructions target
  B200. This task validates the KDA workflow only.
- The initial container did not include PyTorch, `nvcc`, or `ncu`; capture the
  actual runtime versions before comparing numbers.
- A fusion candidate must preserve both observable outputs and FP32 reduction
  semantics. A speedup without both checks is not evidence.

## Candidate directions

1. **Reference baseline:** establish timing and correctness evidence.
2. **One-row Triton reduction:** one program computes one row, keeps the fused
   residual in registers, reduces in FP32, then writes both outputs.
3. **Tile and warp search:** vary hidden-size tile, warps, vector width, and
   tail masking. Keep only shapes whose measured result improves baseline.

## First steps

1. Run validation and baseline benchmark with the prepared V100 environment.
2. Append the measured baseline record to `candidates.jsonl`.
3. Add one isolated candidate with the same `forward` interface.
4. Validate before timing; save any profiler report under `profile/` when
   `ncu` becomes available.
