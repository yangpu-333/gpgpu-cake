"""Rebuild the portable KDA external-skill loader from a fixed KDA commit."""

import argparse
import subprocess
from pathlib import Path


KDA_SHA = "ef6ce617693ef0782b3ecb9f37e39bbf10226a90"
KDA_URL = "https://github.com/NVlabs/kda.git"
PATCH = Path(__file__).with_name("kda-loader.patch")


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new KDA checkout")
    parser.add_argument("--source", default=KDA_URL, help="Git URL or local KDA checkout")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    if not PATCH.is_file():
        parser.error(f"patch missing: {PATCH}")
    source = str(Path(args.source).resolve()) if Path(args.source).exists() else args.source
    run("git", "init", str(output))
    run("git", "-C", str(output), "config", "core.autocrlf", "false")
    run("git", "-C", str(output), "remote", "add", "origin", source)
    run("git", "-C", str(output), "fetch", "--depth", "1", "origin", KDA_SHA)
    run("git", "-C", str(output), "checkout", "--detach", "FETCH_HEAD")
    run("git", "-C", str(output), "apply", "--check", str(PATCH))
    run("git", "-C", str(output), "apply", str(PATCH))
    if not (output / "tools" / "load_external_skill.py").is_file():
        raise RuntimeError("KDA loader missing after patch")
    print(output)


if __name__ == "__main__":
    main()
