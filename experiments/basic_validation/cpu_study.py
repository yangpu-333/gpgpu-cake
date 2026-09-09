"""Reproducible CPU correctness study; no third-party packages or speed claims."""

import argparse
import hashlib
import json
import math
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import run


SEEDS = (0, 7, 41)
DISTRIBUTIONS = ("zeros", "small_integers", "uniform")
LENGTHS = (1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 129, 1024)
BLOCKS = (1, 4, 16, 32, 2048)
SHAPES = ((1, 1, 1), (1, 17, 3), (9, 1, 7), (5, 7, 3),
          (16, 16, 16), (17, 19, 23), (31, 33, 17), (64, 48, 32))
TILES = (1, 2, 4, 8, 128)


def values(rng, length, distribution):
    if distribution == "zeros":
        return [0] * length
    if distribution == "small_integers":
        return [rng.randint(-8, 8) for _ in range(length)]
    return [rng.uniform(-1, 1) for _ in range(length)]


def functional_cases():
    records = []
    for seed in SEEDS:
        for distribution in DISTRIBUTIONS:
            rng = random.Random(seed)
            for length in LENGTHS:
                a, b = values(rng, length, distribution), values(rng, length, distribution)
                before = (a[:], b[:])
                expected = [left + right for left, right in zip(a, b)]
                for block in BLOCKS:
                    actual = run.chunked_add(a, b, block)
                    check = run.compare_values(actual, expected, atol=0, rtol=0)
                    check["inputs_unchanged"] = (a, b) == before
                    check["passed"] &= check["inputs_unchanged"]
                    records.append({"operator": "add", "length": length, "block": block,
                                    "seed": seed, "distribution": distribution, "correctness": check})
            for m, n, k in SHAPES:
                a = [values(rng, k, distribution) for _ in range(m)]
                b = [values(rng, n, distribution) for _ in range(k)]
                before = ([row[:] for row in a], [row[:] for row in b])
                # Integer products/sums here are exact and small enough for binary64.
                # fsum is a more accurate accumulation of rounded float products,
                # not an exact-real dot-product oracle.
                accumulate = math.fsum if distribution == "uniform" else sum
                expected = [accumulate(a[i][p] * b[p][j] for p in range(k))
                            for i in range(m) for j in range(n)]
                tolerance = 1e-12 if distribution == "uniform" else 0
                for tile in TILES:
                    output = run.tiled_matmul(a, b, tile)
                    check = run.compare_values(run.flatten(output), expected,
                                               atol=tolerance, rtol=tolerance)
                    check["shape_correct"] = len(output) == m and all(len(row) == n for row in output)
                    check["inputs_unchanged"] = (a, b) == before
                    check["passed"] &= check["shape_correct"] and check["inputs_unchanged"]
                    records.append({"operator": "matmul", "shape_mnk": [m, n, k], "tile": tile,
                                    "seed": seed, "distribution": distribution,
                                    "reference": "fsum_of_float_products" if tolerance else "exact_integer",
                                    "correctness": check})
    return records


def gate_checks():
    reference = [1.0, 2.0, 3.0]
    injected = {"wrong_tail": [1.0, 2.0, 0.0], "missing_tail": [1.0, 2.0],
                "nan": [1.0, 2.0, math.nan], "infinity": [1.0, 2.0, math.inf]}
    checks, candidates = [], []
    for name, actual in injected.items():
        correctness = run.compare_values(actual, reference)
        checks.append({"name": name, "passed": not correctness["passed"], "checker": correctness})
        candidates.append({"name": name, "kind": "candidate", "correctness": correctness,
                           "median_ms": 0.001})
    good = {"name": "correct", "kind": "candidate", "correctness": {"passed": True}, "median_ms": 1.0}
    untimed = {"name": "untimed", "kind": "candidate", "correctness": {"passed": True}}
    checks.append({"name": "invalid_fast_candidates_excluded",
                   "passed": run.choose_fastest([*candidates, untimed, good]) is good})
    checks.append({"name": "no_valid_candidate_no_selection",
                   "passed": run.choose_fastest([*candidates, untimed]) is None})
    return checks


def accumulation_example():
    a = [[1e16, 1.0, -1e16]]
    b = [[1.0], [1.0], [1.0]]
    original = run.tiled_matmul(a, b, 2)[0][0]
    reordered = run.tiled_matmul([[1e16, -1e16, 1.0]], b, 2)[0][0]
    exact = sum((10**16, 1, -(10**16)))
    return {"original_terms": a[0], "reordered_terms": [1e16, -1e16, 1.0],
            "exact_integer_result": exact, "sequential_float_result": original,
            "reordered_float_result": reordered,
            "original_vs_exact": run.compare_values([original], [exact]),
            "note": "Separate numerical diagnostic, excluded from the bounded functional pass count. "
                    "Reordering a floating-point reduction can change its result. "
                    "This is not evidence that changing tile size alone changes this implementation's order."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    stamp = datetime.now(timezone.utc)
    output = args.output or run.ROOT / "results" / (stamp.strftime("%Y%m%dT%H%M%S%fZ") + "-cpu-study.json")
    if output.exists():
        parser.error("output already exists; choose a new filename")
    started = time.perf_counter()
    records, gates = functional_cases(), gate_checks()
    passed = all(r["correctness"]["passed"] for r in records) and all(c["passed"] for c in gates)
    report = {
        "schema_version": 1, "created_at_utc": stamp.isoformat(),
        "status": "cpu_study_passed" if passed else "cpu_study_failed",
        "scope": "Bounded Python CPU functional checks and synthetic selection checks; "
                 "no kernel timing, GPU, Halide, Poly, or training validation",
        "environment": {"python": sys.version, "platform": platform.platform()},
        "source_sha256": {name: hashlib.sha256((run.ROOT / name).read_bytes()).hexdigest()
                          for name in ("run.py", "cpu_study.py")},
        "settings": {"seeds": SEEDS, "distributions": DISTRIBUTIONS,
                     "lengths": LENGTHS, "blocks": BLOCKS, "shapes_mnk": SHAPES, "tiles": TILES},
        "summary": {op: {"cases": len(rows), "passed": sum(r["correctness"]["passed"] for r in rows),
                         "max_abs_error": max(r["correctness"].get("max_abs_error", 0) for r in rows)}
                    for op in ("add", "matmul") if (rows := [r for r in records if r["operator"] == op])},
        "records": records, "gate_checks": gates,
        "gate_note": "median_ms values in gate checks are synthetic labels, not CPU/GPU measurements; "
                     "this verifies selection filtering, not the GPU launch/timing path",
        "numerical_diagnostic": accumulation_example(),
        "wall_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("report:", output.resolve())
    print(json.dumps({key: report[key] for key in ("summary", "gate_checks", "numerical_diagnostic", "wall_seconds")},
                     ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
