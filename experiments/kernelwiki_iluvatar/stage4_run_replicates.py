"""Run three independent BI-V150 GEMM epilogue/tile rounds per dtype."""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--shapes", default="127x65x33,256x256x256,512x512x512")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    kernel = Path(__file__).with_name("stage4_gemm_epilogue.py")
    manifest = {"runner_sha256": digest(Path(__file__)),
                "kernel_sha256": digest(kernel), "shapes": args.shapes,
                "device": args.device, "runs": [], "summary": [], "complete": False}
    reports = {}
    try:
        for round_number in range(1, 4):
            order = ("float16", "bfloat16") if round_number % 2 else ("bfloat16", "float16")
            for dtype in order:
                seed = 20260928 + (round_number - 1) * 10101
                name = f"{dtype}-round{round_number}"
                report = args.output_dir / f"{name}.json"
                log = args.output_dir / f"{name}.log"
                command = [sys.executable, str(kernel), "--dtype", dtype,
                           "--shapes", args.shapes, "--seed", str(seed),
                           "--device", str(args.device), "--output", str(report)]
                with log.open("w", encoding="utf-8") as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                entry = {"round": round_number, "dtype": dtype, "seed": seed,
                         "command": command, "exit_code": result.returncode,
                         "log": log.name, "log_sha256": digest(log),
                         "report": report.name if report.exists() else None,
                         "report_sha256": digest(report) if report.exists() else None}
                manifest["runs"].append(entry)
                if report.exists():
                    reports[(round_number, dtype)] = json.loads(report.read_text())
                print(name, "exit", result.returncode, flush=True)
                if result.returncode != 0:
                    return 1
        for dtype in ("float16", "bfloat16"):
            for shape in reports[(1, dtype)]["shapes"]:
                mnk = shape["mnk"]
                for index, candidate in enumerate(shape["configs"]):
                    rounds = []
                    for round_number in range(1, 4):
                        target = next(s for s in reports[(round_number, dtype)]["shapes"]
                                      if s["mnk"] == mnk)["configs"][index]
                        if target["config"] != candidate["config"]:
                            raise ValueError("candidate order changed")
                        rounds.append({"round": round_number, "status": target["status"],
                                       "unfused_check": target.get("unfused_check"),
                                       "fused_check": target.get("fused_check"),
                                       "pair_check": target.get("pair_check"),
                                       "timing": target.get("timing"),
                                       "error": target.get("error")})
                    valid = all(r["status"] == "measured" and r["timing"]["speedup"] > 1
                                for r in rounds)
                    manifest["summary"].append({"dtype": dtype, "mnk": mnk,
                                                "config": candidate["config"],
                                                "rounds": rounds,
                                                "valid_and_faster_all_rounds": valid,
                                                "geomean_speedup": math.prod(r["timing"]["speedup"]
                                                        for r in rounds) ** (1 / 3) if valid else None,
                                                "median_fused_us": sorted(r["timing"]["fused_median_us"]
                                                        for r in rounds)[1] if valid else None})
        manifest["complete"] = True
        return 0
    finally:
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
