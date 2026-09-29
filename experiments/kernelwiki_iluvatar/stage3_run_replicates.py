"""Repeat the BI-V150 fused Residual Add RMSNorm forward experiment.

Each run is a separate process and keeps its raw JSON and console log. A
configuration is considered consistently faster only when it passes all
correctness checks and beats the same-process PyTorch baseline in every run.
"""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


SHAPES = "64x768,64x1024,64x4096"
DTYPES = ("float16", "bfloat16")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(reports):
    groups = {}
    for item in reports:
        doc = item["document"]
        if doc.get("status") != "gpu_autotune_passed":
            continue
        dtype = item["dtype"]
        for work in doc["workloads"]:
            shape = "x".join(map(str, work["shape"]))
            key = (dtype, shape)
            baseline = next(r for r in work["candidate_records"] if r["kind"] == "baseline")
            for record in work["candidate_records"]:
                if record["kind"] != "candidate":
                    continue
                config = record["config"]
                name = f"warps{config['num_warps']}-block{config['reduction_tile']}"
                row = groups.setdefault(key, {}).setdefault(name, {
                    "config": config, "rounds": [],
                })
                row["rounds"].append({
                    "round": item["round"], "seed": item["seed"],
                    "status": record["status"],
                    "forward_passed": record.get("forward_correctness", {}).get("passed"),
                    "backward_passed": record.get("backward_correctness", {}).get("passed"),
                    "baseline_median_ms": baseline.get("median_ms"),
                    "candidate_median_ms": record.get("median_ms"),
                    "speedup": record.get("speedup_vs_unfused_pytorch_forward"),
                })
    summary = []
    for (dtype, shape), candidates in sorted(groups.items()):
        entries = []
        for name, data in sorted(candidates.items()):
            rounds = data["rounds"]
            valid = (len(rounds) == 3 and all(
                r["status"] == "measured" and r["forward_passed"] and
                r["backward_passed"] and isinstance(r["speedup"], (int, float)) and
                math.isfinite(r["speedup"]) and r["speedup"] > 0 for r in rounds))
            ratios = [r["speedup"] for r in rounds] if valid else []
            entries.append({
                "name": name, "config": data["config"], "rounds": rounds,
                "all_rounds_valid": valid,
                "geomean_speedup": math.exp(statistics.mean(map(math.log, ratios))) if valid else None,
                "min_speedup": min(ratios) if valid else None,
                "consistently_faster_than_baseline": bool(valid and min(ratios) > 1.0),
            })
        accepted = [entry for entry in entries if entry["consistently_faster_than_baseline"]]
        winner = max(accepted, key=lambda x: x["geomean_speedup"]) if accepted else None
        summary.append({"dtype": dtype, "shape": shape, "candidates": entries,
                        "selected_config": winner["config"] if winner else None,
                        "selection_rule": "highest geometric mean speedup among candidates faster in all 3 independent rounds"})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shapes", default=SHAPES)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--launches", type=int, default=30)
    args = parser.parse_args()
    if args.rounds != 3:
        parser.error("This evidence protocol requires exactly 3 independent rounds")
    if args.output_dir.exists():
        parser.error("output directory exists; select a new path to preserve evidence")
    args.output_dir.mkdir(parents=True)
    script = Path(__file__).with_name("stage3_residual_rmsnorm.py")
    reports = []
    manifest = {"runner_sha256": digest(Path(__file__)), "kernel_script_sha256": digest(script),
                "shapes": args.shapes, "device": args.device, "runs": []}
    for round_index in range(args.rounds):
        for dtype in (DTYPES if round_index % 2 == 0 else tuple(reversed(DTYPES))):
            seed = 20260928 + 101 * round_index
            stem = f"{dtype}-round{round_index + 1}"
            output = args.output_dir / f"{stem}.json"
            log = args.output_dir / f"{stem}.log"
            command = [sys.executable, str(script), "--mode", "gpu", "--device", str(args.device),
                       "--shapes", args.shapes, "--dtype", dtype, "--seed", str(seed),
                       "--atol", "0.03" if dtype == "float16" else "0.04",
                       "--rtol", "0.03" if dtype == "float16" else "0.04",
                       "--warmup", str(args.warmup), "--samples", str(args.samples),
                       "--launches", str(args.launches), "--output", str(output)]
            with log.open("w", encoding="utf-8") as stream:
                try:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                            timeout=600, check=False)
                    exit_code = result.returncode
                except subprocess.TimeoutExpired:
                    stream.write("child process exceeded 600 seconds\n")
                    exit_code = 124
            receipt = {"round": round_index + 1, "dtype": dtype, "seed": seed,
                       "command": command, "exit_code": exit_code,
                       "log": log.name, "log_sha256": digest(log)}
            if output.is_file():
                doc = json.loads(output.read_text(encoding="utf-8"))
                receipt.update(report=output.name, report_sha256=digest(output), status=doc.get("status"))
                reports.append({"round": round_index + 1, "dtype": dtype, "seed": seed,
                                "document": doc})
            manifest["runs"].append(receipt)
            (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                            encoding="utf-8")
    manifest["summary"] = summarize(reports)
    manifest["complete"] = len(reports) == 2 * args.rounds and all(
        r["exit_code"] == 0 and r.get("status") == "gpu_autotune_passed" for r in manifest["runs"])
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"complete={manifest['complete']} workloads={len(manifest['summary'])}")
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
