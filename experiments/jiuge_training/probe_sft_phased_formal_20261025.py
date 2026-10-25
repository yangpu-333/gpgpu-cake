"""Repeat the unchanged guarded CPU/fused thread-switch checks in this container."""
from sft_phased_formal_20261025 import PREP, APPROVAL

if __name__ == '__main__':
    text = (PREP/'probe_sft_phased_acceptance_20261024.py').read_text()
    before = 'sft-phased-acceptance-approved-20261024-01.json'
    assert text.count(before) == 1
    exec(compile(text.replace(before, APPROVAL.name), '<same-live-formal-CPU-admission>', 'exec'),
         {'__name__': '__main__', '__file__': __file__})
