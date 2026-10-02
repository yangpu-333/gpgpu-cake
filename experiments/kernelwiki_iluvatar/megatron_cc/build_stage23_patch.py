"""Serialize a verified adaptation checkout without truncating on Git failures."""
import argparse
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--skill", type=Path, required=True)
args = parser.parse_args()
skill = args.skill.resolve()
prefix = ["git", "-c", "safe.directory=" + skill.as_posix(), "-C", str(skill)]
head = subprocess.check_output(prefix + ["rev-parse", "HEAD"], text=True).strip()
if head != "b6b4301f15e8ce6955a56776690643ce5db369e6":
    raise ValueError("unexpected upstream base")
subprocess.run(prefix + ["add", "-N", "."], check=True)
payload = subprocess.check_output(prefix + ["diff", "--binary", "HEAD"])
if len(payload) < 1000:
    raise ValueError("unexpectedly small adaptation patch")
patch = Path(__file__).resolve().parent.parent / "kernelwiki-iluvatar.patch"
patch.write_bytes(payload)
print(patch, len(payload))
