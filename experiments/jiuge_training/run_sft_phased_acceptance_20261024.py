"""Derive the unchanged SFT runner; bind acceptance to current candidate sources."""
import hashlib
from pathlib import Path
from sft_phased_acceptance_20261024 import PREP, POLICIES, read, replace_exact, sha, verify_sources
import run_sft_pinning_case as original
import builtins

NATIVE_CHECK = original.check
NATIVE_ADDITIONS = original.additions


def check(args):
    NATIVE_CHECK(args)
    files = verify_sources()
    if not args.phased_plan.is_file():
        raise ValueError('Actual CPU/GPU plan required')
    if args.mode == 'checkpoint':
        raise ValueError('Native checkpoint uses the untouched native SFT runner')
    if args.mode == 'formal':
        admission = read(args.admission)
        if (not admission.get('incremental_phased_candidate_accepted')
                or admission.get('incremental_source_sha256') != files
                or admission.get('incremental_baseline') != 'accepted pinned, unbound'):
            raise ValueError('This candidate requires its own full admission')


def additions(args, folder, argv):
    return NATIVE_ADDITIONS(args, folder, argv)+[
        '--phased-plan', str(args.phased_plan), '--phased-policy', POLICIES[args.pinning_arm]]


def derive_generated(text):
    return replace_exact(text, [
        ("parser.add_argument('--case',required=True)", "parser.add_argument('--case',required=True)\n    parser.add_argument('--phased-plan',type=Path,required=True)"),
        ("str(PREP/'sft_pinning_entry.py'),'--evidence'", "str(PREP/'sft_phased_acceptance_entry_20261024.py'),'--evidence'"),
        ("    # This survives SIGKILL/OOM", "    from sft_phased_acceptance_20261024 import verify_sources\n    for name in verify_sources():\n        shutil.copyfile(PREP/name,snapshots/name)\n    # This survives SIGKILL/OOM"),
    ])


def main():
    verify_sources()
    original.check, original.additions = check, additions
    original.compile = lambda text, name, mode: builtins.compile(derive_generated(text), name, mode)
    original.main()


if __name__ == '__main__':
    main()
