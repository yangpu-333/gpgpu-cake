# GDN Decode V100 Agent Contract

- Objective: minimize geometric-mean candidate latency across all 54 official GDN Decode workloads on one Tesla V100.
- Correctness: both output tensors must match the downloaded official reference for every element with atol=rtol=0.01; inputs must remain unchanged.
- External contract: BF16 q/k/v/a/b and output, FP32 state and new_state, four Q/K heads expanded to eight V heads, optional state, and scale=None/0 default behavior must be preserved.
- Allowed implementation: one Python module using only `math`, `torch`, `triton`, and `triton.language`; it must export `triton_candidate(q, k, v, state, A_log, a, dt_bias, b, scale)`.
- Forbidden: modifying the evaluator, reference, inputs, tolerances, evidence, environment, or other files; subprocesses, files, network, dynamic imports, eval, and exec.
- Validation: four fixed smoke workloads followed by all 54 official workloads and three optional-argument branches.
- Promotion: complete correctness and at least 1% lower geometric-mean latency than the current best candidate. A rejected candidate remains in the ledger with its error or performance result.
- Evidence: immutable candidate source, API response metadata, smoke/full JSON reports, parent candidate ID, timing score, and promotion decision.
