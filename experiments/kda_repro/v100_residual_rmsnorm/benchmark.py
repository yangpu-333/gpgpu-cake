"""Median CUDA-event timing for the V100 residual RMSNorm KDA task."""

from __future__ import annotations

import csv
import argparse
import importlib
import statistics
from pathlib import Path

import torch

from src.reference import forward as reference_forward

EPS = 1e-6
SHAPES = ((64, 768), (64, 1024), (64, 4096))
WARMUP = 25
ITERATIONS = 100


def implementation(selection="candidate"):
    if selection == "baseline":
        return "baseline-pytorch", reference_forward
    try:
        candidate = importlib.import_module("src.candidate")
    except ModuleNotFoundError as error:
        if error.name != "src.candidate":
            raise
        raise RuntimeError("Requested candidate is missing") from error
    return "candidate", candidate.forward


def median_ms(kernel, shape: tuple[int, int]) -> float:
    hidden = torch.randn(shape, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(hidden)
    weight = torch.randn(shape[1], device="cuda", dtype=torch.float16)
    for _ in range(WARMUP):
        kernel(hidden, residual, weight, EPS)
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(ITERATIONS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        kernel(hidden, residual, weight, EPS)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("baseline", "candidate"), default="candidate")
    args = parser.parse_args()
    name, kernel = implementation(args.implementation)
    from validate import validate_shape
    torch.manual_seed(20260918)
    for shape in SHAPES:
        validate_shape(shape, kernel)
    csv_path = Path("benchmark.csv")
    rows = []
    for shape in SHAPES:
        rows.append(
            {
                "candidate": name,
                "shape": f"{shape[0]}x{shape[1]}",
                "median_ms": f"{median_ms(kernel, shape):.6f}",
                "iterations": ITERATIONS,
                "device": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "status": "measured",
            }
        )
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
