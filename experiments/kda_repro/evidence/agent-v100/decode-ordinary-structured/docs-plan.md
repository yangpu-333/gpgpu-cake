# Executable agent plan

1. Copy the reviewed V100 candidate into an isolated workspace as candidate 0000 and establish its full 54-workload score.
2. Ask the configured LLM for a short draft before requesting code, following the upstream KDA planning requirement.
3. Give each optimization request the immutable task contract, hardware facts, current best source, aggregate evidence, and recent rejection summaries.
4. Accept a complete source module only. Reject unsafe syntax before import.
5. Run workload indices 0, 10, 25, and 53 as the compile/correctness smoke gate.
6. Run every official workload and optional state/scale branches for candidates that pass smoke.
7. Promote only a fully correct candidate whose candidate-latency geometric mean improves by at least 1%.
8. Append every outcome to `candidates.jsonl`; never delete rejected candidates or overwrite raw reports.
9. Stop at the explicit iteration/API-call budget. Profiling remains disabled until the platform grants performance-counter access.
