"""Install and verify the prepared BI-V150 skill in a KDA checkout."""

import argparse
import subprocess
import sys
from pathlib import Path


EXPECTED_ARCHITECTURE_COUNTS = {"bi-v150": 38, "sm90": 286, "sm100": 379}
EXTENSION_PAGE_PATHS = (
    "sources/experiments/bi-v150-corex42-stage30.md",
    "wiki/kernels/swiglu-heldout-megatron-bi-v150.md",
    "sources/experiments/bi-v150-corex42-stage31.md",
    "wiki/kernels/dense-attention-megatron-bi-v150.md",
    "sources/experiments/bi-v150-corex42-stage32.md",
    "wiki/kernels/dense-attention-heldout-megatron-bi-v150.md",
    "sources/experiments/bi-v150-corex42-stage34.md",
    "wiki/kernels/weighted-swiglu-moe-megatron-bi-v150.md",
    "sources/experiments/bi-v150-corex42-stage35-retry01.md",
    "wiki/kernels/weighted-swiglu-heldout-megatron-bi-v150.md",
)
EXTENSION_BUNDLE = "evidence/bi-v150-corex42-stage30-31"
EXTENSION_EVIDENCE_PATHS = (
    "manifest.json",
    "PROVENANCE.yaml",
    "megatron_bf16/swiglu_scale/evidence/decision.json",
    "megatron_bf16/swiglu_scale/evidence/round-v2/cpu-verify-wide-raw.json",
    "megatron_bf16/swiglu_scale/evidence/round-v2/cpu-verify-deep-raw.json",
    "megatron_bf16/swiglu_scale/evidence/round-v2/performance-wide-004-repair-v2.json",
    "megatron_bf16/swiglu_scale/evidence/round-v2/performance-deep-004-repair-v2.json",
    "megatron_bf16/attention/candidates/003/candidate.py",
    "megatron_bf16/attention/evidence/decision.json",
    "megatron_bf16/attention/evidence/round003/cpu-verify-003.json",
    "megatron_bf16/attention/evidence/round003/performance-main-003-complete.json",
    "megatron_bf16/attention/verify_failed_rounds.py",
)


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8")


def query_paths(skill, architecture):
    response = run(
        sys.executable, str(skill / "scripts" / "query.py"),
        "--architecture", architecture, "--paths-only", "--limit", "1000",
    )
    return {line.replace("\\", "/") for line in response.stdout.splitlines()}


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
        if architecture == "bi-v150":
            missing = set(EXTENSION_PAGE_PATHS) - actual
            if missing:
                raise RuntimeError(f"stage30/31/32/34/35 BI-V150 query pages missing: {sorted(missing)}")
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
                     "evidence/bi-v150-corex42-stage27/megatron_bf16/evidence/rmsnorm-stage27/decision.json",
                     "sources/experiments/bi-v150-corex42-stage28.md",
                     "evidence/bi-v150-corex42-stage28/megatron_bf16/evidence/rmsnorm-stage28/decision.json",
                     "sources/experiments/bi-v150-corex42-stage29.md",
                     "wiki/kernels/swiglu-megatron-bi-v150.md",
                     "evidence/bi-v150-corex42-stage29/megatron_bf16/swiglu/candidates/004/candidate.py",
                     "evidence/bi-v150-corex42-stage29/megatron_bf16/evidence/swiglu-stage29/decision.json"):
        if not (project_skill / relative).is_file():
            raise RuntimeError(f"native Megatron loop evidence missing: {relative}")
    for relative in (*EXTENSION_PAGE_PATHS,
                     *(f"{EXTENSION_BUNDLE}/{path}" for path in EXTENSION_EVIDENCE_PATHS)):
        if not (project_skill / relative).is_file():
            raise RuntimeError(f"stage30/31 Megatron loop evidence missing: {relative}")
    for relative in ("manifest.json", "PROVENANCE.yaml",
                     "megatron_bf16/attention_scale/evidence/decision.json",
                     "megatron_bf16/attention_scale/evidence/cpu-verify-wide-raw.json",
                     "megatron_bf16/attention_scale/evidence/cpu-verify-deep-raw.json",
                     "megatron_bf16/attention_scale/evidence/performance-wide-003-complete02.json",
                     "megatron_bf16/attention_scale/evidence/performance-deep-003-complete02.json"):
        if not (project_skill / "evidence/bi-v150-corex42-stage32" / relative).is_file():
            raise RuntimeError(f"stage32 frozen Attention evidence missing: {relative}")
    for relative in ("manifest.json", "PROVENANCE.yaml",
                     "megatron_bf16/moe_weighted/evidence/decision.json",
                     "megatron_bf16/moe_weighted/evidence/performance-002.json",
                     "megatron_bf16/moe_weighted/evidence/performance-003.json",
                     "megatron_bf16/moe_weighted/evidence/comparison-native-001.json",
                     "megatron_bf16/moe_weighted/evidence/cpu-verify-004-raw.json",
                     "megatron_bf16/moe_weighted/evidence/cpu-verify-004-confirmation-raw.json",
                     "megatron_bf16/moe_weighted/evidence/confirmation/performance-004.json",
                     "megatron_bf16/moe_weighted/candidates/004/candidate.py"):
        if not (project_skill / "evidence/bi-v150-corex42-stage34" / relative).is_file():
            raise RuntimeError(f"stage34 weighted MoE evidence missing: {relative}")
    for relative in ("manifest.json", "PROVENANCE.yaml",
                     "megatron_bf16/moe_scale_recovery01/evidence/decision-retry01-02.json",
                     "megatron_bf16/moe_scale_recovery01/evidence/completed-formal-import-retry01-02.json",
                     "megatron_bf16/moe_scale_recovery01/evidence/raw-http-backup-retry01-01-receipt.json",
                     "megatron_bf16/moe_scale_recovery01/evidence/long-primary-diagnostic-import-01.json",
                     "megatron_bf16/moe_scale_recovery01/models-retry01/long/evidence/confirmation/performance-004.json",
                     "megatron_bf16/moe_scale_recovery01/models-retry01/experts/evidence/performance-004.json"):
        if not (project_skill / "evidence/bi-v150-corex42-stage35-retry01" / relative).is_file():
            raise RuntimeError(f"stage35 fresh heldout evidence missing: {relative}")
    print("KDA project-level skill integration verified")


if __name__ == "__main__":
    main()
