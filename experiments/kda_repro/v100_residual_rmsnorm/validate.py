"""Correctness validation for the V100 residual RMSNorm KDA task."""

from __future__ import annotations

import importlib
import argparse
import json
from pathlib import Path

import torch

from src.reference import forward as reference_forward

EPS = 1e-6
SHAPES = ((64, 768), (64, 1024), (64, 4096))
ATOL = 2e-3
RTOL = 2e-3


def implementation(selection="candidate"):
    if selection == "baseline":
        return "baseline-pytorch", reference_forward
    try:
        candidate = importlib.import_module("src.candidate")
    except ModuleNotFoundError as error:
        if error.name != "src.candidate":
            raise
        raise RuntimeError("Requested candidate is missing; use --implementation baseline explicitly") from error
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

    inputs = (hidden, residual, weight)
    # Random vector-Jacobian products check gradients, not just their finiteness.
    # Exercise both outputs independently as well as their combined contribution.
    for mode in ("normalized", "residual", "both"):
        grad_y = torch.randn_like(output) if mode != "residual" else torch.zeros_like(output)
        grad_r = torch.randn_like(residual_out) if mode != "normalized" else torch.zeros_like(residual_out)
        actual = torch.autograd.grad((output, residual_out), inputs, (grad_y, grad_r), retain_graph=True)
        expected = torch.autograd.grad((expected_output, expected_residual), inputs, (grad_y, grad_r), retain_graph=True)
        for name, got, want in zip(("hidden", "residual", "weight"), actual, expected):
            torch.testing.assert_close(got, want, atol=ATOL, rtol=RTOL, msg=f"{mode}/{name} gradient mismatch")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(20260918)
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("baseline", "candidate"), default="candidate")
    args = parser.parse_args()
    name, kernel = implementation(args.implementation)
    for shape in SHAPES:
        validate_shape(shape, kernel)
    print(json.dumps({"status": "validation_passed", "candidate": name, "shapes": SHAPES}))


if __name__ == "__main__":
    main()
