"""Reuse the frozen resume runner with a weak-reference-only observer repair."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    data = (PREP / 'run_numa_validation_case.py').read_bytes()
    if hashlib.sha256(data).hexdigest() != '25e98afe955287b536e5f39f1488659f941c53602316b990a7998643b7e508b9':
        raise ValueError('Frozen validation derivation changed')
    text = data.decode().replace('numa_entry.py', 'numa_observation_v2_entry.py')
    before = "if args.steps!=3 or args.seq_length!=2048 or args.capture_windows or args.capture_inputs or args.reference:"
    if text.count(before) != 1:
        raise ValueError('Resume guard derivation changed')
    text = text.replace(before, before.replace('if ', "if args.validation_stage!='resume' or ", 1))
    # Add this actual derivation to the source tuple embedded in the frozen runner.
    needle = "'run_numa_validation_case.py'):"
    if text.count(needle) != 1:
        raise ValueError('Source tuple derivation changed')
    text = text.replace(needle, "'run_numa_validation_case.py','run_numa_resume_v2_case.py'):")
    exec(compile(text, str(PREP / 'run_numa_resume_v2_case.derived.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
