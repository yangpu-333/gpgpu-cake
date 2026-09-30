"""Assemble the BI-V150 KernelWiki extension on its fixed upstream commit.

The patch contains only local edits and experiment evidence. Fetching upstream
at build time keeps the original NVIDIA corpus and its third-party assets under
their own provenance and terms.
"""

import argparse
import subprocess
from pathlib import Path


UPSTREAM_SHA = "b6b4301f15e8ce6955a56776690643ce5db369e6"
UPSTREAM_URL = "https://github.com/mit-han-lab/KernelWiki.git"
PATCH = Path(__file__).with_name("kernelwiki-iluvatar.patch")


def run(*args):
    subprocess.run(args, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New skill directory")
    parser.add_argument("--source", default=UPSTREAM_URL,
                        help="Upstream Git URL or a local checkout for offline testing")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        parser.error(f"output already exists: {target}")
    if not PATCH.is_file():
        parser.error(f"patch missing: {PATCH}")
    source = str(Path(args.source).resolve()) if Path(args.source).exists() else args.source

    run("git", "-c", "core.longpaths=true", "init", str(target))
    run("git", "-C", str(target), "config", "core.longpaths", "true")
    run("git", "-C", str(target), "config", "core.autocrlf", "false")
    run("git", "-C", str(target), "remote", "add", "origin", source)
    run("git", "-C", str(target), "fetch", "--depth", "1", "origin", UPSTREAM_SHA)
    run("git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD")
    run("git", "-C", str(target), "apply", "--whitespace=nowarn", "--check", str(PATCH))
    run("git", "-C", str(target), "apply", "--whitespace=nowarn", str(PATCH))
    if not (target / "SKILL.md").is_file():
        raise RuntimeError("assembled skill has no SKILL.md")
    print(target)


if __name__ == "__main__":
    main()
