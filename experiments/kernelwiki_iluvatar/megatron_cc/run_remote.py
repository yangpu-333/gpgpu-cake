"""Serial BI-V150 runs; preserve every process command, result, stdout and stderr."""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_id")
    parser.add_argument("phase", choices=("smoke", "screen", "screen-v3", "final"))
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{4}", args.candidate_id):
        parser.error("candidate id must have four digits")
    candidate = ROOT / "candidates" / args.candidate_id / "candidate.py"
    folder = ROOT / "evidence" / (args.candidate_id + "-" + args.phase)
    folder.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    base = [sys.executable, "-u", str(ROOT / "benchmark.py"),
            "--megatron-root", "/private/atrex-megatron/src/megatron-lm", "--candidate", str(candidate)]
    tasks = [("operator-smoke", ["--mode", "operator", "--smoke"])]
    if args.phase in ("screen", "screen-v3"):
        tasks = [("operator-23001", ["--mode", "operator"]),
                 ("primary-23001", ["--mode", "model"])]
    elif args.phase == "final":
        tasks = [("operator-" + str(seed), ["--mode", "operator", "--seed", str(seed)])
                 for seed in (23002, 23003, 23004)]
        tasks += [("primary-" + str(seed), ["--mode", "model", "--seed", str(seed)])
                  for seed in (23002, 23003, 23004)]
        tasks += [("holdout-23005", ["--mode", "model", "--seed", "23005", "--layers", "2",
                   "--hidden", "256", "--heads", "4", "--sequence", "17", "--vocab", "512"]),
                  ("autocast-23006", ["--mode", "model", "--seed", "23006", "--autocast"])]
    manifest = {"started_utc": datetime.now(timezone.utc).isoformat(), "candidate_id": args.candidate_id,
                "phase": args.phase, "gpu_visible": "0", "runs": []}
    with (ROOT / ".gpu-experiment.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for name, flags in tasks:
            output = folder / (name + ".json")
            command = base + flags + ["--output", str(output)]
            record = {"name": name, "command": command,
                      "started_utc": datetime.now(timezone.utc).isoformat()}
            with (folder / (name + ".stdout")).open("w") as out, (folder / (name + ".stderr")).open("w") as err:
                try:
                    process = subprocess.run(command, stdout=out, stderr=err, timeout=600)
                    record["exit_code"] = process.returncode
                except subprocess.TimeoutExpired:
                    record["exit_code"] = 124
                    record["error"] = "600 second process timeout"
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["runs"].append(record)
            (folder / "commands.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(name, record["exit_code"], flush=True)
            if record["exit_code"] != 0:
                print("Stopped after a concrete correctness/error failure; evidence preserved", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
