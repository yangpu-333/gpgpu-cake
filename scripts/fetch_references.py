"""Download optional public references and verify the recorded SHA256 (stdlib only)."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


RESEARCH = Path(__file__).resolve().parents[1] / "research"
GROUPS = {
    "cake": {"papers/CAKE-2608.12629.pdf", "code/cake-ir.zip"},
    "atrex": {"papers/ATREX-2607.14541.pdf", "code/atrex-kernel-agent.zip"},
    "flashinfer": {
        "code/flashinfer-pr-4262.diff",
        "code/flashinfer-pr-4274.diff",
        "code/flashinfer-pr-4638.diff",
    },
}


def check_file(path, entry):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    if size != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
        raise ValueError(f"size/SHA256 mismatch: {path}; existing file preserved")


def fetch(entry, verify_only=False):
    destination = (RESEARCH / entry["path"]).resolve()
    if not destination.is_relative_to(RESEARCH.resolve()):
        raise ValueError("manifest path must stay inside research/")
    if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
        raise ValueError("invalid SHA256 in manifest")
    if not isinstance(entry["bytes"], int) or entry["bytes"] <= 0:
        raise ValueError("invalid file size in manifest")
    if urlparse(entry["url"]).scheme != "https":
        raise ValueError("reference URL must use HTTPS")

    if destination.exists():
        check_file(destination, entry)
        print(f"VERIFIED {entry['path']}")
        return
    if verify_only:
        raise FileNotFoundError(f"missing reference: {entry['path']}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=".reference-", suffix=".part", dir=destination.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            request = Request(entry["url"], headers={"User-Agent": "GPGPU-CAKE-reference-fetch/1.0"})
            with urlopen(request, timeout=30) as response:
                size = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    size += len(block)
                    if size > entry["bytes"]:
                        raise ValueError("download exceeds recorded size; review the source version")
                    output.write(block)
        check_file(temporary, entry)
        # An exclusive destination prevents overwriting a file from another run.
        created = False
        try:
            with destination.open("xb") as output:
                created = True
                with temporary.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
        except BaseException:
            if created:
                destination.unlink(missing_ok=True)
            raise
        print(f"DOWNLOADED {entry['path']}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=[*GROUPS, "all"], default="cake")
    parser.add_argument("--verify-only", action="store_true", help="check local files without network access")
    args = parser.parse_args(argv)
    wanted = set().union(*GROUPS.values()) if args.group == "all" else GROUPS[args.group]
    manifest = json.loads((RESEARCH / "sources" / "download-manifest.json").read_text(encoding="utf-8-sig"))
    entries = {entry["path"]: entry for entry in manifest["files"]}
    failures = 0
    for path in sorted(wanted):
        try:
            fetch(entries[path], verify_only=args.verify_only)
        except (OSError, ValueError, KeyError) as error:
            print(f"FAILED {path}: {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
