# Executable GDN Prefill Agent Plan

1. Start from the reviewed sequential Triton implementation and change one bounded code region per iteration.
2. Prefer algebraic simplification, row scheduling, launch geometry, and memory-access changes that preserve
   the recurrence order.
3. Apply the model's exact replacements and run the static source policy.
4. Reject immediately on any of four representative smoke workloads.
5. Validate all 100 official workloads and optional-state/empty-sequence branches.
6. In one process, collect 21 alternating timing samples for the current best and new candidate.
7. Promote only after full correctness and at least 1% paired geometric-mean improvement; otherwise record
   the rejection reason and retain the current best.
