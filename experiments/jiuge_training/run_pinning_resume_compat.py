"""Derive an untimed native resume with only the common-state API adapter."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED = '57df7dc75e8d6d25b3c88a4b7e89a441cdfbbe912f099cf689db575fb68f127f'


def main():
    source = (PREP / 'run_pinning_semantics.py').read_bytes()
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError('Frozen semantic runner changed')
    text = source.decode()
    replacements = [
        ("str(PREP/'pinning_semantics_entry.py'),'--evidence'",
         "str(PREP/'pinning_resume_compat_entry.py'),'--evidence'"),
        ("    command += ['--pinning-arm',args.pinning_arm",
         "    command += ['--checkpoint-compat-receipt',str(folder/'rank-{rank}/checkpoint-compat.json')]\n    command += ['--pinning-arm',args.pinning_arm"),
        ("'pinning_full_audit.py','run_pinning_semantics.py'):",
         "'pinning_full_audit.py','run_pinning_semantics.py','checkpoint_load_compat.py','pinning_resume_compat_entry.py','run_pinning_resume_compat.py'):")]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('Resume launch derivation no longer exact')
        text = text.replace(before, after)
    exec(compile(text, str(PREP / 'run_pinning_resume_compat.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': str(PREP / 'run_pinning_semantics.py')})


if __name__ == '__main__':
    main()
