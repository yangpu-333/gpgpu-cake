# GDN Prefill V100 Agent Task Contract

- Optimize `gdn_prefill_qk4_v8_d128_k_last` for the existing V100/sm70 environment.
- Preserve the exact function signature and return `(output_bf16, new_state_fp32)`.
- Preserve sequential time recurrence, variable sequence lengths, empty-sequence behavior, optional state,
  default scale behavior, input immutability, and the official `atol=rtol=0.01` correctness rule.
- Use only the imports already allowed by the controller and export exactly one `triton_candidate`.
- A candidate must pass four smoke workloads and all 100 official workloads.
- Promotion requires at least 1% lower geometric-mean latency than the current best in the same paired run.
- Do not use released final contest solutions as input.
