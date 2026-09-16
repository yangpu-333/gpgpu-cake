"""Combine per-rank runtime capture reports without interpreting them as benchmarks."""

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def shape_key(record):
    module = record.get("module", {}).get("class_name")
    input_info = record.get("input", {})
    return (
        module,
        tuple(input_info.get("shape") or ()),
        input_info.get("dtype"),
        input_info.get("device"),
    )


def summarize_reports(input_dir):
    input_dir = Path(input_dir)
    reports = []
    for path in sorted(input_dir.glob("rmsnorm-capture-*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        reports.append((path, payload))
    groups = {}
    for path, report in reports:
        for record in report.get("records", []):
            key = shape_key(record)
            group = groups.setdefault(key, {"count": 0, "event_ms": [], "ranks": set(), "statuses": {}})
            group["count"] += 1
            group["ranks"].add(str(report.get("environment", {}).get("rank")))
            status = record.get("status", "unknown")
            group["statuses"][status] = group["statuses"].get(status, 0) + 1
            timing = record.get("sampled_forward_event_ms")
            if isinstance(timing, (int, float)) and math.isfinite(timing) and timing > 0:
                group["event_ms"].append(float(timing))
    workloads = []
    for (class_name, shape, dtype, device), group in sorted(groups.items(), key=lambda item: str(item[0])):
        timings = group["event_ms"]
        workloads.append({
            "class_name": class_name,
            "input_shape": list(shape),
            "dtype": dtype,
            "device": device,
            "observed_calls": group["count"],
            "ranks": sorted(group["ranks"]),
            "record_statuses": group["statuses"],
            "sampled_event_ms": ({"median": statistics.median(timings), "min": min(timings), "max": max(timings)}
                                 if timings else None),
        })
    target_record_count = sum(group["count"] for group in groups.values())
    if target_record_count:
        status = "target_calls_captured"
    elif reports:
        status = "no_target_module_calls"
    else:
        status = "no_capture_reports_found"
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "operator": "residual_add_rmsnorm_capture_summary",
        "status": status,
        "source_reports": [str(path) for path, _ in reports],
        "target_record_count": target_record_count,
        "workloads": workloads,
        "interpretation": "shape-discovery report only; sampled event intervals are not a training benchmark",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite capture evidence")
    report = summarize_reports(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print("status:", report["status"])
    print("report:", args.output.resolve())
    for workload in report["workloads"]:
        print("workload:", workload["class_name"], workload["input_shape"], "calls:", workload["observed_calls"])
    return 0 if report["status"] == "target_calls_captured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
