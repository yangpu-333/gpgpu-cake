"""Apply the frozen contract to independent final runs; keep raw samples intact."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite decision")
    folder = ROOT / "evidence" / (args.candidate_id + "-final")
    files = sorted(folder.glob("*.json"))
    reports = {p.stem: json.loads(p.read_text()) for p in files if p.name != "commands.json"}
    expected = {"operator-23002", "operator-23003", "operator-23004",
                "primary-23002", "primary-23003", "primary-23004", "holdout-23005", "autocast-23006"}
    contract = json.loads((ROOT / "contract.json").read_text())
    gates = contract["promotion"]
    hashes = {"candidate_sha256": sha(ROOT / "candidates" / args.candidate_id / "candidate.py"),
              "harness_sha256": sha(ROOT / "benchmark.py"), "contract_sha256": sha(ROOT / "contract.json")}
    complete = expected.issubset(reports)
    identical = all(all(r.get(k) == v for k, v in hashes.items()) for r in reports.values())
    passed = complete and all(reports[name]["status"] == "passed" for name in expected)
    primary = [reports[name]["model"]["training"] for name in sorted(expected)
               if name.startswith("primary-") and name in reports and "training" in reports[name].get("model", {})]
    ratios = [r["speedup"] for r in primary]
    geomean = math.exp(statistics.mean(math.log(v) for v in ratios)) if len(ratios) == 3 else None
    memory = {}
    for name, r in reports.items():
        t = r.get("model", {}).get("training")
        if t:
            peaks = {k: statistics.median(v) for k, v in t["incremental_peak_bytes"].items()}
            memory[name] = peaks["candidate"] / peaks["native"] - 1
    robustness = {name: reports[name].get("model", {}).get("training", {}).get("speedup")
                  for name in ("holdout-23005", "autocast-23006") if name in reports}
    checks = {"all_required_runs_complete": complete, "identical_frozen_sources": identical,
              "all_correctness_and_native_compatibility_pass": passed,
              "three_primary_rounds_each_faster": len(ratios) == 3 and all(v > 1 for v in ratios),
              "primary_geomean_at_least_1_01": geomean is not None and geomean >= gates["primary_wall_speedup_geomean_min"],
              "holdout_and_autocast_at_least_0_98": len(robustness) == 2 and all(v is not None and v >= gates["holdout_and_autocast_speedup_min"] for v in robustness.values()),
              "temporary_peak_memory_regression_at_most_5_percent": len(memory) == 5 and all(v <= gates["memory_regression_max_fraction"] for v in memory.values())}
    result = {"candidate_id": args.candidate_id, "contract": contract["version"], **hashes,
              "decision": "promote_for_tested_scope" if all(checks.values()) else "reject_or_experimental",
              "checks": checks, "primary_speedups": ratios, "primary_geomean": geomean,
              "primary_median_ms": [r["median_ms"] for r in primary], "robustness_speedups": robustness,
              "temporary_peak_memory_fraction": memory,
              "operator_case_summaries": {name: r.get("case_summary") for name, r in reports.items() if name.startswith("operator-")},
              "scope": "BI-V150 CoreX 4.2, fixed Megatron local backend, FP32 residual fusion, representative synthetic GPT steps with SGD; not production or low-precision residual training",
              "evidence_folder": str(folder.relative_to(ROOT))}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
