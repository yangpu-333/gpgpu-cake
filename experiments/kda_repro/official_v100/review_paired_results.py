"""Review repeated paired reports and append an auditable promotion decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("latencies must be finite positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def review_reports(reports: list[tuple[Path, dict]], min_improvement: float) -> dict:
    if len(reports) < 3:
        raise ValueError("at least three paired reports are required")
    baseline_values, candidate_values = [], []
    expected_uuids = None
    baseline_hashes, candidate_hashes = set(), set()
    candidate_wins = 0
    report_summaries = []
    for path, report in reports:
        if report.get("status") != "complete":
            raise ValueError(f"incomplete report: {path}")
        workloads = report.get("workloads", [])
        uuids = [workload.get("uuid") for workload in workloads]
        if not workloads or (expected_uuids is not None and uuids != expected_uuids):
            raise ValueError("paired reports must contain the same non-empty workload sequence")
        expected_uuids = uuids
        baseline_hashes.add(report.get("baseline_source_sha256"))
        candidate_hashes.add(report.get("source_sha256"))
        run_baseline, run_candidate = [], []
        for workload in workloads:
            candidates = workload.get("candidates", {})
            baseline, candidate = candidates.get("baseline", {}), candidates.get("candidate", {})
            if baseline.get("status") != "correct" or candidate.get("status") != "correct":
                raise ValueError(f"incorrect paired result for {workload.get('uuid')}")
            run_baseline.append(float(baseline["median_ms"]))
            run_candidate.append(float(candidate["median_ms"]))
            candidate_wins += candidate["median_ms"] < baseline["median_ms"]
        baseline_values.extend(run_baseline)
        candidate_values.extend(run_candidate)
        baseline_geo, candidate_geo = geometric_mean(run_baseline), geometric_mean(run_candidate)
        report_summaries.append({
            "path": str(path), "workloads": len(workloads),
            "baseline_geomean_ms": baseline_geo,
            "candidate_geomean_ms": candidate_geo,
            "speedup": baseline_geo / candidate_geo,
        })
    if len(baseline_hashes) != 1 or None in baseline_hashes:
        raise ValueError("baseline source hash differs across reports")
    if len(candidate_hashes) != 1 or None in candidate_hashes:
        raise ValueError("candidate source hash differs across reports")
    baseline_geo, candidate_geo = geometric_mean(baseline_values), geometric_mean(candidate_values)
    speedup = baseline_geo / candidate_geo
    meets_threshold = candidate_geo < baseline_geo * (1.0 - min_improvement)
    return {
        "reports": report_summaries,
        "report_count": len(reports),
        "workloads_per_report": len(expected_uuids),
        "paired_measurements": len(candidate_values),
        "baseline_source_sha256": next(iter(baseline_hashes)),
        "candidate_source_sha256": next(iter(candidate_hashes)),
        "baseline_geomean_ms": baseline_geo,
        "candidate_geomean_ms": candidate_geo,
        "paired_speedup": speedup,
        "paired_improvement_percent": (1.0 - 1.0 / speedup) * 100.0,
        "candidate_wins": candidate_wins,
        "required_improvement_percent": min_improvement * 100.0,
        "meets_threshold": meets_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--reports", required=True, nargs="+", type=Path)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 < args.min_improvement < 1:
        raise SystemExit("min improvement must be between zero and one")
    reports = []
    report_hashes = []
    for path in args.reports:
        resolved = path.resolve()
        reports.append((resolved, json.loads(resolved.read_text())))
        report_hashes.append(hashlib.sha256(resolved.read_bytes()).hexdigest())
    review = review_reports(reports, args.min_improvement)
    ledger = args.workspace.resolve() / "candidates.jsonl"
    if not ledger.exists():
        raise SystemExit(f"missing ledger: {ledger}")
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    candidate_records = [record for record in records
                         if record.get("candidate_id") == args.candidate_id]
    if not candidate_records:
        raise SystemExit(f"candidate absent from ledger: {args.candidate_id}")
    previous = candidate_records[-1]
    if previous.get("source_sha256") and previous["source_sha256"] != review["candidate_source_sha256"]:
        raise SystemExit("candidate source hash does not match ledger")
    parent_id = previous.get("parent_id")
    parent_records = [record for record in records if record.get("candidate_id") == parent_id]
    if parent_records and parent_records[-1].get("source_sha256") and \
            parent_records[-1]["source_sha256"] != review["baseline_source_sha256"]:
        raise SystemExit("baseline source hash does not match parent ledger entry")
    status = "promoted" if review["meets_threshold"] else "demoted"
    decision = {
        "candidate_id": args.candidate_id,
        "parent_id": parent_id,
        "status": status,
        "score_ms": review["candidate_geomean_ms"],
        "paired_baseline_ms": review["baseline_geomean_ms"],
        "paired_speedup": review["paired_speedup"],
        "paired_improvement_percent": review["paired_improvement_percent"],
        "source_sha256": review["candidate_source_sha256"],
        "reason": ("repeated paired evidence meets promotion threshold" if review["meets_threshold"]
                   else "repeated paired evidence does not meet promotion threshold"),
        "review_report_sha256": report_hashes,
    }
    already_recorded = (previous.get("status") == status and
                        previous.get("review_report_sha256") == report_hashes)
    if not already_recorded:
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "review_recorded" if not already_recorded else "review_already_recorded",
                      "decision": decision, "review": review}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
