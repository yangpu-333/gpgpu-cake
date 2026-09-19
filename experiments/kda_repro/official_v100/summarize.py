"""Validate and summarize raw official-task evidence without third-party packages."""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report.get("status") != "complete":
        raise SystemExit(f"report is not complete: {report.get('status')}")
    if not report.get("workloads"):
        raise SystemExit("report contains no workloads")

    speedups = []
    winners = Counter()
    by_batch = defaultdict(list)
    max_output_error = 0.0
    max_state_error = 0.0
    for workload in report["workloads"]:
        rejected = [name for name, value in workload["candidates"].items()
                    if value["status"] != "correct"]
        if rejected:
            raise SystemExit(f"{workload['uuid']} contains rejected candidates: {rejected}")
        winner = workload["winner"]
        winners[winner] += 1
        base = workload["candidates"]["vectorized"]["median_ms"]
        chosen = workload["candidates"][winner]
        speedup = base / chosen["median_ms"]
        speedups.append(speedup)
        by_batch[str(workload["batch"])].append(speedup)
        max_output_error = max(max_output_error, chosen["max_abs"][0])
        max_state_error = max(max_state_error, chosen["max_abs"][1])

    summary = {
        "status": "validated_summary",
        "workloads": len(report["workloads"]),
        "branches": len(report.get("branches", [])),
        "dataset_revision": report["dataset"]["revision"],
        "winners": dict(winners),
        "speedup_geomean": math.exp(sum(math.log(x) for x in speedups) / len(speedups)),
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "max_output_abs_error": max_output_error,
        "max_state_abs_error": max_state_error,
        "by_batch": {key: {
            "count": len(values),
            "speedup_geomean": math.exp(sum(math.log(x) for x in values) / len(values)),
            "speedup_min": min(values),
            "speedup_max": max(values),
        } for key, values in sorted(by_batch.items(), key=lambda item: int(item[0]))},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
