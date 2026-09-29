"""Probe only the first minimal BI-V150 GPU tensor allocation."""

import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

start = time.monotonic()
print("BEGIN import_torch", flush=True)
import torch

print("END import_torch", torch.__version__, flush=True)
print("BEGIN allocate_empty", f"cuda:{args.device}", flush=True)
tensor = torch.empty(16, device=f"cuda:{args.device}")
print("ALLOC_OK", tuple(tensor.shape), "seconds", round(time.monotonic() - start, 3), flush=True)
