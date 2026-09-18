# Executable plan

1. Record `python`, `torch`, CUDA runtime, driver, device name, and compute
   capability in the first run record.
2. Run `validate.py` with the baseline implementation and retain its console
   output in `runs/`.
3. Run `benchmark.py`; append each median latency to `benchmark.csv`.
4. Update the `baseline-pytorch` JSON record from `planned` to `validated` or
   `rejected`, with links to the two artifacts.
5. Only then create `src/candidate.py`. Run the same validation and benchmark
   commands before recording a promotion decision.
