# KDA Optimization Draft — GDN Decode on Tesla V100

**Parent candidate:** `0000-seed` · **Baseline geomean:** `0.264630284 ms` · **Promotion bar:** ≤ `0.262` ms (≥1% faster) with full correctness (atol=rtol=0.01).

---

## 1. Current Baseline: What It Does

The seed is a correctness-first eager PyTorch implementation of one **gated delta-rule (GDN) decode step**. Working per `(batch b, head h)` on a `128×128` FP32 state matrix `S`:

- Expand 4 Q/K heads → 8 V heads via `repeat_interleave(2)`.
- Per-head scalar gate `g = exp(-exp(A_log) · softplus(a + dt_bias))` and scalar `beta = sigmoid(b)`.
- Decay: `old = g · S`.
- Recall: `old_v[i] = Σ_j old[i,j]·k[j]` (matvec `old @ k`).
- Delta write: `new_v = beta·v + (1-beta)·old_v`.
- Rank-1 update: `updated[i,j] = old[i,j] − k[j]·old_v[i] + k[j]·new_v[i]`.
- Readout: `out[i] = Σ_j updated[i,j]·q[j]` scaled by `scale` (default `1/√128`).

**Cost profile:** the state tensor `[B, 8, 128, 128]` FP32 dominates memory traffic. The eager path materializes ~5–6 full state-sized intermediates (`old`, `old*kf`, two rank-1 terms, `updated`), so it makes many bandwidth passes over the largest tensor. This is a **memory-bandwidth-bound** problem — the decode step has no matmul big enough to exploit compute.

> Note: exact input tensor ranks (e.g. whether `A_log`/`beta`/`g` are `[B,8]` per-head scalars) will be confirmed against the actual workloads before coding. The algebra below assumes per-head scalar `g` and `beta`, which matches the seed's broadcasting.

---

## 2. Key Algebraic Simplification (drives every candidate)

The rank-1 update collapses:

```
updated = old − k·old_v + k·new_v
        = old + k·(new_v − old_v)
        = old + k ⊗ (beta·(v − old_v))     since new_v − old_v = beta·(v − old_v)
```

Define `dv[i] = beta·(v[i] − g·(S@k)[i])`. Then the readout closes in **two reductions plus one dot product**, avoiding materializing `updated` for the output:

```
Sk[i] = Σ_j S[i,j]·k[j]          (matvec)
Sq[i] = Σ_j S[i,j]·q[j]          (matvec)
kq    = Σ_j k[j]·q[j]            (scalar dot)
dv[i] = beta·(v[i] − g·Sk[i])
out[i]   = g·Sq[i] + dv[i]·kq
new_S[i,j] = g·S[i,j] + k[j]·dv[i]
```

This means a fused kernel touches the state **exactly once for read and once for write** — the theoretical bandwidth floor for this op.

---

## 3. V100 (sm_70) Risks

- **No compute win from tensor cores.** Decode is a single-timestep rank-1 update → matvec/outer-product only. FP16/BF16 tensor cores give nothing here; the win is purely from cutting memory traffic. Don't chase `tl.dot`.
- **FP32 state = 4× BF16 footprint.** State read/write is the bottleneck; every extra pass costs full bandwidth (~900 GB/s ceiling).
- **Low occupancy on small batch.** Grid of `B×8` programs may be too small; must tile the 128 rows to keep SMs fed for small `B` workloads.
- **Triton-on-Volta feature gaps.** Stick to loads/stores, elementwise math, and `tl.sum` reductions; avoid newer intrinsics that may not lower cleanly on sm_70.
- **Precision.** Compute all of `exp/softplus/sigmoid`, both matvecs, and readout in **FP32** (BF16 inputs upcast on load). This preserves the reference's `.float()` semantics and should clear atol=rtol=0.01 comfortably.
- **`scale` falsy-default trap.** Reference uses `scale = scale or 1/√128`, so `scale=0` **must** fall back to the default. Preserve this exactly.
- **Optional `state=None`.** Must allocate a zero state; then `Sk=Sq=0`, `dv=beta·v`, `new_S = k⊗dv`, `out = dv·kq`. Verify this branch explicitly.
- **Input immutability.** Never write in-place into inputs; `new_state` must be a fresh allocation.
- **Register pressure.** A full `128×128` FP32 tile is 64 KB — cannot live entirely in registers; use row-tiling with `j=128` held as a vector, and consider `num_warps` tuning.

---

## 4. Ranked Candidate Directions

**C1 — Single fused Triton kernel, algebra-simplified (primary).**
Grid over `(b, h)` (or `(b, h, row-tile)`). Each program loads a row-block of `S` once, computes `Sk`/`Sq` reductions and `kq`, writes `new_S = g·S + k⊗dv` and `out`. One read + one write of state. Expected the biggest win (targets the ~3× traffic reduction). **Highest priority.**

**C2 — Row-tiled variant of C1 for occupancy.**
Same math, but split the 128 rows across multiple programs (`BLOCK_M ∈ {16,32,64}`) so small-batch workloads still saturate SMs. Sweep `BLOCK_M` and `num_warps ∈ {2,4,8}`. Layer on top of C1 once correct.

**C3 — Head-fused loads.**
Because 4 Q/K heads expand to 8 V heads, adjacent V-head pairs share `q/k`. Fuse the head pair into one program to reuse the loaded `q/k`/`kq` and improve L2 reuse. Secondary; apply only if C1/C2 leave headroom.

**C4 — Memory-layout / vectorization tuning.**
Ensure contiguous `128`-wide `j` loads (coalesced), vectorized BF16→FP32 conversion, and store `out` as BF16 directly. Minor, but cheap to fold into C1.

**C5 (fallback) — Eager fusion without Triton.**
Rewrite the eager path using the C2-algebra with fewer intermediates (`torch.einsum`/`baddbmm`) as a safety net if Triton lowering misbehaves on sm_70. Lower ceiling but low risk.

---

## 5. First Steps

1. Confirm exact input shapes/dtypes and the `scale`/`state=None` branch semantics against real workloads (indices 0, 10, 25, 53 span the range).
2. Lock in the C2 algebra (`Sk`, `Sq`, `kq`, `dv`, closed-form `out`, `new_S`).
3. Draft **C1** as a single fused kernel: grid `(B, 8)`, `j` as a length-128 vector, FP32 accumulation, BF16 output store.
4. Wire the Python wrapper: default-scale handling (`or`-fallback), zero-state allocation, contiguity/stride passing, fresh `new_state`.
5. Get C1 through the smoke gate before any tuning.
6. Add **C2** row-tiling + `num_warps`/`BLOCK_M` sweep only after C1 is correct.

---

## 6. Exact Validation Gates

- **Syntax/safety pre-check:** module imports only `math`, `torch`, `triton`, `triton.language`; exports `triton_candidate(q, k, v, state, A_log, a, dt_bias, b, scale)`; no subprocess/file/network/dynamic-import/eval/exec.
- **Smoke gate:** workloads **0, 10, 25, 53** must compile and match reference (both output tensors) at **atol=rtol=0.01**.
- **Full gate:** all **54 official workloads** pass correctness.
- **Optional branches:** `state=None`, `scale=None`, and `scale=0` (must default to `1/√128`) all pass.
- **Input immutability:** inputs bitwise-unchanged after the call.
- **Timing:** candidate-latency **geometric mean** over the 54 workloads.

## 7. Promotion Evidence to Record

- Immutable candidate source + `source_sha256`.
- Parent candidate ID (`0000-seed`).
- API response metadata for the generating request.
- Smoke JSON report and full 54-workload JSON report (raw, never overwritten).
- Measured geomean latency and per-workload timings; promote only if ≤ `0.262` ms (≥1% under baseline) **and** fully correct.
- Promotion decision appended to `candidates.jsonl`; rejected candidates retained with their error/perf result.

---

**Expected outcome:** C1's 1-read/1-write fusion should clear the 1% bar decisively (memory-traffic reduction ≈3×); C2 tuning captures the remaining small-batch occupancy gains. Profiling stays disabled until performance-counter access is granted — decisions rely on wall-clock geomean only.