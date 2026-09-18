# V100 Residual Add + RMSNorm KDA task template

This is a **generic KDA workflow smoke task** for one NVIDIA V100.  It is not
an MLSys FlashInfer contest submission and its measurements must not be
compared with the B200 contest leaderboard.

Create an isolated working copy after preparing `~/kda-repro/.venv`:

```bash
cd /home/huids25/gpgpu-cake
bash experiments/kda_repro/scripts/bootstrap_v100_residual_rmsnorm.sh
cd ~/kda-repro/workspaces/v100-residual-rmsnorm
~/kda-repro/.venv/bin/python validate.py
~/kda-repro/.venv/bin/python benchmark.py
```

To run both commands and append the corresponding evidence record:

```bash
cd /home/huids25/gpgpu-cake
bash experiments/kda_repro/scripts/run_v100_residual_rmsnorm_baseline.sh
```

The template follows KDA's required evidence loop.  `src/reference.py` is the
baseline. A future candidate goes in `src/candidate.py` and exports a
`forward(hidden, residual, weight, eps)` function returning `(output,
residual_out)`.  Never promote it until `validate.py` passes and a measured
result is recorded in both `benchmark.csv` and `candidates.jsonl`.
