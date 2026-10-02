# Higher-Gain Experiment Plan — TP=1 Fused Cross-Entropy Extension of Candidate 0003

## Objective and invariants

Seek a materially higher complete-step wall-clock improvement over pinned native Megatron on BI-V150 by adding a TP=1, zero-label-smoothing fused vocabulary cross-entropy to the **optimized model instance's** `compute_language_model_loss`, while retaining candidate 0003's residual-add + RMSNorm fusion **exactly** (same math, epsilon semantics, two-kernel backward, process-local compiled-launch cache, and autocast/low-precision native fallback).

Held fixed and NOT touched by generated code (contract `megatron_commit 5be9626…`, `frozen_base_harness_sha256 360e57d1…`):
- Baseline: unmodified `get_gpt_layer_local_spec(normalization='RMSNorm')` + `get_bias_dropout_add(training=True, fused=False)`.
- Model: 4-layer, H512, 8 heads, seq128, micro-batch2, vocab1024, FP32, SGD, dropout0, `layernorm_epsilon 1e-5`. Holdout: 2-layer H256, seq17, vocab512.
- Synthetic input distribution, RNG discipline (no seed hard-coding), tolerances (CE `atol 3e-4`, `rtol 1e-3` on CPU float64 copies), optimizer work, and wall-clock timing (`time.perf_counter` around consecutive complete steps, sync only at block boundaries; warmup5, samples12, steps_per_sample10, 3 independent processes; speedup = baseline median / candidate median).
- TP/config/layout fallback and all correctness/promotion checks are owned by the **frozen external adapter/verifier**, not by generated code. Generated code only provides `cross_entropy(logits[S,B,V], target[S,B]) -> float32 loss[S,B]` plus the unchanged fused residual/RMSNorm API, byte-identical to 0003.

## Why extend CE now

Parent 0003 improved native by only ~1.08% (3-process FP32 complete-step geomean 1.010846). Further residual-only micro-tuning has limited headroom. The diagnostic warm profile (overhead-heavy legacy CUDA profiler; `~19540 cudaEventRecord` for 5 steps, negative/implausible self-device aggregates) is used **only as a call-pattern clue, never as promotion throughput or exclusive GPU fraction**: it shows 9 native RMSNorm forward calls and 3 all_reduce calls per step, and a native CE path (`_VocabParallelCrossEntropy` + `native_vocab_parallel_cross_entropy`) composed of many small aten ops (max, in-place subtract, exp, sum, log, index, div, masked gradient scatter, mul). The old cold cProfile total is NOT reused as a hot-path timing claim. The hypothesis: fusing the per-row CE forward and backward into one kernel each removes launch/materialization overhead from these many small ops, adding a second independent speedup source on top of the retained residual/RMSNorm fusion.

## Native CE semantics to reproduce exactly (TP=1 only)

From `native_cross_entropy.py` specialized to `world_size=1`, `label_smoothing=0`, `vocab_start_index=0`, `vocab_end_index=V`:
1. FP32 path: logits are cast to FP32 and the FP32 buffer is **overwritten in place** — first `logits -= max` (row max), then `exp`, then `div_` by `sum_exp` so the FP32 input buffer ends holding softmax probabilities. This observable forward mutation MUST be preserved. For low-precision (fp16/bf16) inputs the original input is cast to a separate FP32 copy and the **original low-precision tensor is left unchanged**.
2. Loss (FP32): `log(sum(exp(logits - max))) - (target_logit - max)`, i.e. stable row logsumexp minus the shifted target logit.
3. Out-of-range targets (including -100) are **NOT ignore_index**: with a single partition the only mask is `target < 0 | target >= V`; for masked rows the predicted term is 0 (loss = logsumexp) and backward gradient is plain `softmax * dloss` with no `-1` at any class.
4. Backward: `grad = softmax`; subtract `1.0` at the (unmasked) target class; multiply elementwise by `grad_output`; the saved probabilities buffer is overwritten. Autograd casts the returned gradient back to the original input dtype.

## Conservative kernel design (no code in this plan)

- **Forward: one row-wise Triton kernel.** Grid over S*B rows, `BLOCK = next_power_of_2(V)` with masking, `num_warps=4`, consistent with the observed CoreX 4.2 / Triton 2.1 surface. One pass computes row max, the shifted logsumexp and the target-logit term in FP32; it writes the FP32 loss[S,B] and writes the normalized softmax probabilities back into the FP32 logits buffer so the native forward mutation is reproduced. Use `1.0 / tl.sqrt`-style explicit reciprocals and constexpr arithmetic (no host `triton.cdiv` inside JIT bounds) per the recorded BI-V150 Triton compatibility notes. Save only the probabilities buffer (already resident), the target indices, and the out-of-range mask for backward — **no extra FP32 probabilities matrix is allocated**, to stay within the two-resident-model temporary-memory gate.
- **Backward: one row-wise Triton kernel.** Reads saved probabilities, subtracts 1.0 at the unmasked target class, multiplies by `grad_output`, writes the gradient; autograd casts to the original input dtype. No distributed all_reduce (TP=1).
- **Launch/compile reuse:** reuse 0003's process-local `_COMPILED_KERNELS` pattern keyed by device, dtype, V, block, and pointer alignment. Cache **compiled code + launch metadata only — never tensors, results, gradients, or device buffers**, and never cache across-call results. Keep the current stream and all required specializations.
- **Dispatch:** the generated function handles only the TP=1, supported-layout, supported-dtype range; everything else (TP>1, unsupported layouts/dtypes, autocast where rounding must match native) is handed to the captured actual native fallback owned by the frozen adapter. No NVIDIA instruction-equivalence or distributed-speedup claims.

## Correctness tests (owned by frozen verifier; plan states coverage)

All numerical comparisons on CPU float64 copies at `atol 3e-4 / rtol 1e-3`, outside timed blocks. Across CE shapes `[128,2,1024]`, `[17,2,512]`, `[7,3,769]`, `[1,1,7]`, `[2,1,1]` and dtypes fp32/fp16/bf16:
- Unreduced loss[S,B] vs native.
- All logits gradients vs native (independent autograd comparison).
- Observable native forward input mutation on FP32 (buffer becomes softmax) AND unchanged original low-precision input.
- Shape and dtype of outputs/gradients.
- Boundary / out-of-range labels (negative, `>= V`, -100) producing logsumexp loss and softmax*dloss gradient (no class `-1`).
- Extreme-logit distributions: scale .001, scale 1, scale 100, zero ties, `+10000` constant offset, dominating `±1000` class — verifying numerical stability of the max-shift.
- Nonuniform signed-and-zero upstream gradient vectors.
- Native-invalid cases checked separately on CPU for matching NaN / signed-infinity patterns (never counted as finite correctness).
- Native fallback selected and bit-matching for unsupported layouts and under autocast.
- No standalone CE microbenchmark (native mutates inputs, so repeated-input CE timing is invalid); CE speed is evidenced only through complete-step wall-clock.

## Promotion gates (complete-step)

Promote only if ALL hold:
- All correctness checks (residual/RMSNorm from 0003 and new CE) pass.
- **Three independent primary FP32 complete-step rounds, each with extended/native ratio > 1.**
- Primary 3-process wall-speedup **geomean ≥ 1.05**.
- **Normalized independent current-best control ≥ 1.02**: incremental ratio `(extended/native) / (current-best 0003/native)`, each arm paired to its own contemporaneous native process (an explicitly normalized independent comparison, not a three-arm same-process comparison). `current_best_candidate_sha256 b426ef0a…`.
- Holdout (2-layer) and bf16-autocast complete-step speedups each **≥ 0.98** (no-regression gates; autocast logs actual fused input/output dtypes and uses native fallback for CE+RMSNorm there — it is a compatibility gate, not a BF16 training-speed claim).
- Temporary peak-memory regression over the two-resident-model gate **≤ 5%**; excludes single-model total-memory claims.

If the geomean, per-round, current-best, no-regression, or memory gates are not met, **reject or retain as experimental** — a complete loop may correctly end in rejection. No throughput claim is made from the diagnostic profiler, and no production / TP>1 / multi-GPU / TE / low-precision-training speedup is claimed unless independently evidenced.

## Flow and sequencing

Use `prompts/basic-flow.md` as the starter; iterate in the separate task workspace (not this reference repo), writing the draft to `docs/draft.md` there and generated outputs under `runs/`/`outputs/`/`profile/`. No code, no shell, no edits, and no credentials in this step — this is plan-only.
