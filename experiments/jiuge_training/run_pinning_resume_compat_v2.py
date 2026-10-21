"""Same frozen semantic runner with the versioned load-order adapter."""
from pathlib import Path
import hashlib

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    payload = (PREP / 'run_pinning_resume_compat.py').read_bytes()
    if hashlib.sha256(payload).hexdigest() != 'f29e980797bb4425a29e6ddbd0ab430c6edc8d1409cb1440fda74f1a11c82649':
        raise ValueError('Prior compatibility runner changed')
    text = payload.decode()
    before = 'pinning_resume_compat_entry.py'
    if text.count(before) != 2:
        raise ValueError('Unexpected entry derivation')
    text = text.replace(before, 'pinning_resume_compat_entry_v2.py')
    before = "'run_pinning_resume_compat.py'):"
    after = "'run_pinning_resume_compat.py','run_pinning_resume_compat_v2.py','checkpoint_hdo_load_compat.py'):"
    if text.count(before) != 1:
        raise ValueError('Unexpected source snapshot derivation')
    text = text.replace(before, after)
    exec(compile(text, str(PREP / 'run_pinning_resume_compat_v2.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
