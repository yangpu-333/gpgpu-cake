"""Correctness validation for the V100 residual RMSNorm KDA task."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import torch

from src.reference import forward as reference_forward

EPS = 1e-6
SHAPES = ((64, 768), (64, 1024), (64, 4096))
ATOL = 2e-3
RTOL = 2e-3


def implementation():
    try:
        candidate = importlib.import_module("src.candidate")
    except ModuleNotFoundError as error:
        if error.name != "src.candidate":
            raise
        return "baseline-pytorch", reference_forward
    return "candidate", candidate.forward


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, atol=ATOL, rtol=RTOL):
        maximum = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{name} differs from reference; max_abs_error={maximum}")


def validate_shape(shape: tuple[int, int], kernel) -> None:
    hidden = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    residual = torch.randn(shape, device="cuda", dtype=torch.float16, requires_grad=True)
    weight = torch.randn(shape[1], device="cuda", dtype=torch.float16, requires_grad=True)

    output, residual_out = kernel(hidden, residual, weight, EPS)
    expected_output, expected_residual = reference_forward(hidden, residual, weight, EPS)
    assert_close("residual_out", residual_out, expected_residual)
    assert_close("output", output, expected_output)
    if not bool(torch.isfinite(output).all() and torch.isfinite(residual_out).all()):
        raise AssertionError("non-finite forward output")

    (output.float().sum() + residual_out.float().sum()).backward()
    for name, tensor in (("hidden", hidden), ("residual", residual), ("weight", weight)):
        if tensor.grad is None or not bool(torch.isfinite(tensor.grad).all()):
            raise AssertionError(f"missing or non-finite gradient for {name}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260918)
    name, kernel = implementation()
    for shape in SHAPES:
        validate_shape(shape, kernel)
    print(json.dumps({"status": "validation_passed", "candidate": name, "shapes": SHAPES}))


if __name__ == "__main__":
    main()
