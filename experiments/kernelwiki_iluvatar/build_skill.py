"""Assemble the BI-V150 KernelWiki extension on its fixed upstream commit.

The ordered patches contain local edits and experiment evidence. Their SHA256
manifest preserves the original base patch and binds the additive extension. Fetching upstream
at build time keeps the original NVIDIA corpus and its third-party assets under
their own provenance and terms.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


UPSTREAM_SHA = "b6b4301f15e8ce6955a56776690643ce5db369e6"
UPSTREAM_URL = "https://github.com/mit-han-lab/KernelWiki.git"
PATCH = Path(__file__).with_name("kernelwiki-iluvatar.patch")
PATCH_MANIFEST = Path(__file__).with_name("skill-patches.json")


def ordered_patches():
    """Require the shipped base and extension bytes in their declared order."""
    manifest = json.loads(PATCH_MANIFEST.read_text(encoding="utf-8"))
    if manifest["upstream_sha"] != UPSTREAM_SHA:
        raise RuntimeError("patch manifest upstream does not match the fixed commit")
    expected_names = ["kernelwiki-iluvatar.patch", "kernelwiki-iluvatar-stage30-31.patch",
                      "kernelwiki-iluvatar-stage32.patch", "kernelwiki-iluvatar-stage34.patch"]
    rows = manifest["patches"]
    if [row["file"] for row in rows] != expected_names:
        raise RuntimeError("unexpected patch order")
    paths = []
    for row in rows:
        path = PATCH.parent / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise RuntimeError("patch SHA256 differs: " + row["file"])
        paths.append(path)
    return paths


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
    patches = ordered_patches()
    source = str(Path(args.source).resolve()) if Path(args.source).exists() else args.source

    run("git", "-c", "core.longpaths=true", "init", str(target))
    run("git", "-C", str(target), "config", "core.longpaths", "true")
    run("git", "-C", str(target), "config", "core.autocrlf", "false")
    run("git", "-C", str(target), "remote", "add", "origin", source)
    run("git", "-C", str(target), "fetch", "--depth", "1", "origin", UPSTREAM_SHA)
    run("git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD")
    for patch in patches:
        run("git", "-C", str(target), "apply", "--whitespace=nowarn", "--check", str(patch))
        run("git", "-C", str(target), "apply", "--whitespace=nowarn", str(patch))
    if not (target / "SKILL.md").is_file():
        raise RuntimeError("assembled skill has no SKILL.md")
    print(target)


if __name__ == "__main__":
    main()
