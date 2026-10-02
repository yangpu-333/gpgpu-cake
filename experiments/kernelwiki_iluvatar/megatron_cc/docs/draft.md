# KDA Plan — Residual-Add + RMSNorm fused candidate vs pinned Megatron native local backend (BI-V150 / CoreX 4.2)

## 0. Scope and references actually read
- Contract: `megatron_cc/contract.json` (version `megatron-native-biv150-cc-v1`, megatron_commit `5be9626709af2722333bf54797c954c09edeada3`).
- Forward kernel snapshot: `references/stage3_residual_rmsnorm.py` (`exp-bi-v150-corex42-stage3`).
- Backward kernel snapshot: `references/stage13_backward.py` (`exp-bi-v150-corex42-stage13`).
- Skill pages: `kernel-residual-rmsnorm-bi-v150`, `kernel-residual-rmsnorm-backward-bi-v150`, and `references/iluvatar-migration-ledger.md`.
- **Limitation:** the third listed reference snapshot — the Megatron native-baseline harness — could not be opened under this retry's read-only configuration. I did not guess its contents. Baseline semantics below come from `contract.json` and the `exp-bi-v150-corex42-stage21` descriptions on both kernel pages. The implementation turn must open that snapshot before coding; if it stays unavailable, treat the harness as frozen and read baseline behavior only from verifier observations.

## 1. Objective restated
Produce `fused(x, residual, weight, epsilon) -> (normalized, residual_sum)` that drops into the attention `self_attn_bda -> pre_mlp_layernorm` call site of an **unmodified** pinned Megatron local spec (`get_gpt_layer_local_spec(normalization='RMSNorm')` + `get_bias_dropout_add(training=True, fused=False)`), and verify full training-step throughput. The candidate must not touch the harness, verifier, tolerances, input distribution, baseline, or vendor packages.

## 2. Baseline change that invalidates the old adapter (must-fix invariants)
1. **Epsilon is now a real parameter.** Native `TransformerConfig` default is `1e-5` (contract `layernorm_epsilon: 0.00001`), and `operator_checks.epsilons` exercises BOTH `1e-5` and `1e-6`. The previous adapter hard-coded `1e-6`. The candidate MUST accept and thread the caller's `epsilon` into both the forward reduction and the backward `inv` computation. The stage3/stage13 kernels already take `EPSILON`/`EPS` as `constexpr`, but stage13's module-level `EPSILON = 1e-6` and stage3's `--epsilon 1e-6` default are the exact trap to remove — epsilon must come from the call, never from a constant.
2. **Return dtypes must match native, not input.** Norm is `torch.nn.RMSNorm` in this CoreX build. A float32 weight can make RMSNorm emit **float32 output from a lower-precision input**. The stage3 reference forces the normalized output back to `x.dtype` (`.to(x.dtype)` in `reference_forward`); that is only correct against the old adapter, NOT necessarily against the real native module. The candidate must reproduce the native return dtype of BOTH `normalized` and `residual_sum`, determined from verifier observations — never assume `output.dtype == x.dtype`.
3. **Preserve native BDA residual conversion + rounding.** `get_bias_dropout_add(..., fused=False)` with dropout=0, bias=None reduces to a residual add, but with native dtype promotion/rounding. Contract tests mixed pairs `(fp16,fp32)` and `(bf16,fp32)` and a bf16-autocast context in which the stage21 evidence shows the fusion point **still received FP32 tensors**. The observable `residual_sum` must be bit-faithful to native: same promotion order and same output dtype (likely FP32 when residual is FP32), checked by the contract's "exact residual sum and matching return dtypes" and "inputs unchanged" checks.
4. **FP32 reduction, no reassociation shortcut.** Keep `fp32_accumulate` for sum-of-squares (stage3 already upcasts); do not assume reassociation is legal (IR marks it `not_assumed_legal`).
5. **Backward contract.** `grad_x` must equal `grad_residual` bitwise; `grad_weight` via FP32 partials reduced across rows; optional incoming `grad_residual_out` (`h`) added only when present. Keep `atol=rtol=0.03` behavior; backward epsilon must also be the caller's value.
6. **`tl.rsqrt` is absent** in this Triton build — keep `1.0 / tl.sqrt(...)`.

## 3. Ranked candidate improvements (smallest first)

### C1 — SMALLEST FIRST CANDIDATE: epsilon-correct, dtype-faithful thin adapter (FP32 primary path)
- Wrap the already-validated stage3 forward + stage13 backward in a single `torch.autograd.Function` exposing `fused(x, residual, weight, epsilon)`.
- Thread the caller `epsilon` through forward and backward; delete every `1e-6`/`1e-5` constant.
- Match native return dtypes exactly by reading verifier dtype observations before selecting the output/residual dtype — do not cast to `x.dtype` by default.
- Scope dispatch to the contract model's FP32 primary context (`hidden_size=512`, captured RMSNorm input `[8,2,256]`-class rows), where stage21 showed full-step median ratios **1.0183–1.0228** vs the local baseline. Everything outside the validated contract falls back to the native op.
- **Why smallest:** reuses two already-correctness-proven kernels; the only new logic is the epsilon plumbing fix and the dtype-faithfulness fix — the two things that actually broke. No new kernel, no shape sweep, no autocast claims.
- **Risk:** low on correctness; throughput edge is thin (~1.8–2.3%), so it may legitimately fail the `primary_wall_speedup_geomean_min: 1.01` bar on some runs — an honest reject is an acceptable loop outcome per the contract.

### C2 — mixed-dtype residual-add fidelity `(bf16/fp16 input, fp32 residual)`
- The stage3 kernel adds `X + RESIDUAL` assuming one dtype; mixed pairs need the add done in the native promotion order with the native output dtype for `residual_sum`.
- Safest first move: for mixed-dtype inputs, **fall back to native** for the residual add / residual_sum (or perform the add in FP32 and store `residual_sum` in the native-observed dtype), keeping only the normalization fused. Promote to a fused mixed-dtype kernel only after the verifier confirms the exact native dtype/rounding.
- **Risk:** this is the single most likely correctness failure (dtype + rounding mismatch), so gate it behind verifier dtype evidence.

### C3 — validated shape-based dispatch + warp selection
- Add shape dispatch over the contract's `shapes` `[[16,256],[256,512],[7,769],[17,1537]]` with warp counts chosen ONLY after per-shape correctness. Evidence: `num_warps=1` failed the normalized-output check on 64×768 and 64×1024 in all three runs, but passed at 7×769 — so never pick warps without re-checking the target shape. Native fallback outside the documented supported range.
- **Risk:** medium; purely a performance/selection layer, guarded by correctness.

### C4 — fused backward for full-step throughput (defer)
- Only if C1–C3 pass and a measured full-step deficit traces to backward launch overhead. Keep the two-kernel stage13 path; it already passed 72 case-rounds at `atol=rtol=0.03`.
- **Risk:** higher surface area for little expected full-step gain; do last.

### Explicitly NOT pursued now
- Any BF16/FP16-autocast speedup claim. Stage21 shows the autocast context fed FP32 to this fusion point and ran at **0.9188–0.9279** of baseline. The `holdout_and_autocast_speedup_min: 0.98` promotion bar means a fused autocast path would likely REGRESS; prefer native fallback in autocast unless the candidate proves ≥0.98 with evidence.

## 4. Exact semantic risks (checklist for the implementation turn)
- [ ] Epsilon sourced from the call for BOTH forward and backward; `1e-5` and `1e-6` both pass.
- [ ] `normalized` and `residual_sum` dtypes == native-observed dtypes (not forced to `x.dtype`).
- [ ] `residual_sum` bit-exact to native add (promotion order + rounding) for `(fp16,fp32)` and `(bf16,fp32)`.
- [ ] FP32 sum-of-squares accumulation preserved; no reassociation shortcut.
- [ ] `grad_x == grad_residual` bitwise; `grad_weight` FP32-partial reduced; optional `h` honored.
- [ ] Inputs unmolested (contract "inputs unchanged"); no RNG seed / value hard-coding; no cross-call caching.
- [ ] `1.0 / tl.sqrt` (no `tl.rsqrt`).
- [ ] Native fallback for any dtype/shape outside the validated contract.
- [ ] Harness, verifier, tolerances, baseline, vendor packages untouched.

## 5. Validation / promotion gates (from contract)
- All correctness checks pass (outputs, gradients, finite, exact residual sum, matching dtypes, inputs unchanged, independent autograd).
- Operator timing: `torch.cuda.Event`, warmup=10, 15 samples × 30 launches, alternating order, µs/call.
- Training: `perf_counter` around complete steps (zero_grad→fwd→loss→bwd→opt.step), warmup=5, 12 samples × 10 steps, 3 independent processes.
- Promote only if: 3 primary rounds each faster, primary wall speedup geomean ≥ 1.01, holdout+autocast ≥ 0.98, memory regression ≤ 5% (two-resident-model peak scope).
- **No timings or verification claims until external verifier evidence arrives.** A clean reject is a valid loop outcome.

## 6. Draft location and next step
- Per KDA basic flow, write this as `docs/draft.md` inside a separate implementation workspace (not this reference repo), then convert to an executable plan. First implementation turn builds C1 only, submits for external GPU verification, and records candidate SHA256 + parent/feedback before considering C2+.
