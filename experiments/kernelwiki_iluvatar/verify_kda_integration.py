"""Install and verify the prepared BI-V150 skill in a KDA checkout."""

import argparse
import subprocess
import sys
from pathlib import Path


EXPECTED_ARCHITECTURE_COUNTS = {"bi-v150": 25, "sm90": 286, "sm100": 379}


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8")


def query_paths(skill, architecture):
    response = run(
        sys.executable, str(skill / "scripts" / "query.py"),
        "--architecture", architecture, "--paths-only", "--limit", "1000",
    )
    return set(response.stdout.splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kda", type=Path, required=True)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    kda = args.kda.resolve(strict=True)
    skill = args.skill.resolve(strict=True)
    loader = kda / "tools" / "load_external_skill.py"
    if not loader.is_file():
        parser.error(f"KDA loader not found: {loader}")
    if not (skill / "SKILL.md").is_file():
        parser.error(f"prepared skill missing SKILL.md: {skill}")
    base = [sys.executable, str(loader), "--source", str(skill),
            "--name", "kernelwiki-iluvatar"]
    install = list(base)
    if args.replace:
        install.append("--replace")
    print(run(*install).stdout.strip())
    print(run(*base, "--check").stdout.strip())
    project_skill = kda / ".claude" / "skills" / "kernelwiki-iluvatar"
    for architecture, expected in EXPECTED_ARCHITECTURE_COUNTS.items():
        actual = query_paths(project_skill, architecture)
        if len(actual) != expected:
            raise RuntimeError(
                f"{architecture}: expected {expected} query paths, got {len(actual)}"
            )
        print(f"{architecture}: {len(actual)} paths")
    stage14 = project_skill / "sources" / "experiments" / "bi-v150-corex42-stage14.md"
    if not stage14.is_file():
        raise RuntimeError("stage14 BI-V150 evidence page missing from KDA project skill")
    for relative in ("sources/experiments/bi-v150-corex42-stage23.md",
                     "wiki/kernels/residual-rmsnorm-megatron-bi-v150.md",
                     "evidence/bi-v150-corex42-stage23/candidates/0003/candidate.py",
                     "sources/experiments/bi-v150-corex42-stage24.md",
                     "wiki/kernels/cross-entropy-megatron-bi-v150.md",
                     "evidence/bi-v150-corex42-stage24/higher_gain/candidates/0006/candidate.py",
                     "sources/experiments/bi-v150-corex42-stage25.md",
                     "evidence/bi-v150-corex42-stage25/megatron_bf16/candidates/0008/candidate.py",
                     "evidence/bi-v150-corex42-stage25/megatron_bf16/candidates/0009/candidate.py",
                     "evidence/bi-v150-corex42-stage25/megatron_bf16/evidence/decision.json",
                     "sources/experiments/bi-v150-corex42-stage26.md",
                     "evidence/bi-v150-corex42-stage26/megatron_bf16/gradnorm_candidates/001/candidate.py",
                     "evidence/bi-v150-corex42-stage26/megatron_bf16/evidence/gradnorm-stage26/decision.json",
                     "sources/experiments/bi-v150-corex42-stage27.md",
                     "evidence/bi-v150-corex42-stage27/megatron_bf16/evidence/rmsnorm-stage27/affine-006.cu",
                     "evidence/bi-v150-corex42-stage27/megatron_bf16/evidence/rmsnorm-stage27/decision.json"):
        if not (project_skill / relative).is_file():
            raise RuntimeError(f"native Megatron loop evidence missing: {relative}")
    print("KDA project-level skill integration verified")


if __name__ == "__main__":
    main()
