"""Verify and summarize the three BI-V150 backward experiment JSON files."""

import hashlib
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "evidence" / "stage13-20260929"
SOURCE = ROOT / "stage13_backward.py"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    paths = [EVIDENCE / f"round-{seed}.json" for seed in (13001, 13002, 13003)]
    docs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    source_hash = sha256(SOURCE)
    for seed, doc in zip((13001, 13002, 13003), docs):
        assert doc["status"] == "passed"
        assert doc["script_sha256"] == source_hash
        assert doc["arguments"]["seed"] == seed
        assert doc["gpu"] == docs[0]["gpu"]
        assert (doc["torch"], doc["triton"]) == (docs[0]["torch"], docs[0]["triton"])
        assert len(doc["cases"]) == 24

    groups = {}
    for doc in docs:
        for case in doc["cases"]:
            key = (case["dtype"], case["grad_residual_out"], *case["shape"])
            assert case["passed"]
            assert case["checks"]["grad_x_equals_grad_residual"]
            assert case["checks"]["inputs_unchanged"]
            assert all(item["passed"] for item in case["checks"]["autograd"].values())
            assert math.isfinite(case["speedup_vs_analytical_pytorch"])
            groups.setdefault(key, []).append(case)
    assert len(groups) == 24
    assert all(len(cases) == 3 for cases in groups.values())

    summary = []
    for (dtype, mode, rows, hidden), cases in sorted(groups.items()):
        ratios = [case["speedup_vs_analytical_pytorch"] for case in cases]
        summary.append({
            "dtype": dtype, "grad_residual_out": mode, "shape": [rows, hidden],
            "all_rounds_correct": True,
            "all_rounds_faster": all(ratio > 1 for ratio in ratios),
            "speedups": ratios,
            "geomean_speedup": math.exp(statistics.mean(math.log(ratio) for ratio in ratios)),
            "min_speedup": min(ratios),
            "candidate_median_ms": statistics.median(case["median_ms"]["candidate"] for case in cases),
            "baseline_median_ms": statistics.median(case["median_ms"]["analytical_pytorch"] for case in cases),
            "max_abs_autograd_error": max(
                check["max_abs_error"]
                for case in cases for check in case["checks"]["autograd"].values()
            ),
        })
    manifest = {
        "gpu": docs[0]["gpu"], "torch": docs[0]["torch"], "triton": docs[0]["triton"],
        "source_sha256": source_hash,
        "baseline": docs[0]["baseline"],
        "rounds": [
            {"seed": seed, "file": path.name, "sha256": sha256(path)}
            for seed, path in zip((13001, 13002, 13003), paths)
        ],
        "case_count": sum(len(doc["cases"]) for doc in docs),
        "all_correct": True,
        "all_faster": all(item["all_rounds_faster"] for item in summary),
        "summary": summary,
    }
    output = EVIDENCE / "manifest.json"
    output.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"cases={manifest['case_count']} groups={len(summary)} all_faster={manifest['all_faster']}")
    print(f"speedup_range={min(item['geomean_speedup'] for item in summary):.3f}.."
          f"{max(item['geomean_speedup'] for item in summary):.3f}")
    print(output)


if __name__ == "__main__":
    main()
