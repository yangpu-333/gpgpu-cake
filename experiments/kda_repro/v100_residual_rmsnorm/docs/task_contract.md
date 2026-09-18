# Task contract: V100 residual add + RMSNorm forward

- **Objective:** implement and optimize a forward kernel that computes
  `residual_out = hidden + residual` and
  `output = residual_out * rsqrt(mean(residual_out^2) + eps) * weight`.
- **Inputs:** CUDA FP16 tensors `hidden` and `residual` of shape `[rows,
  hidden_size]`, an FP16 `weight` of shape `[hidden_size]`, and scalar
  `eps = 1e-6`.
- **Outputs:** FP16 `output` and FP16 `residual_out`, both `[rows,
  hidden_size]`.
- **Correctness:** outputs must be finite; `residual_out` must agree with the
  elementwise sum and `output` with the FP32-accumulated reference within
  `atol=2e-3, rtol=2e-3`. Gradients from a scalar loss must exist and be finite.
- **Constraints:** run on one Tesla V100 (sm70), use the isolated PyTorch CUDA
  environment, and keep the reference implementation independent from any
  candidate. A candidate may use PyTorch, Triton, or CUDA only after its
  dependencies are recorded.
- **Validation:** `~/kda-repro/.venv/bin/python validate.py`.
- **Evaluation:** `~/kda-repro/.venv/bin/python benchmark.py`.
- **Promotion:** a candidate passes validation for every listed shape, has a
  recorded median CUDA-event time, and improves the matching baseline by at
  least 5% without changing the contract. Otherwise record it as revised or
  rejected in `candidates.jsonl`.
