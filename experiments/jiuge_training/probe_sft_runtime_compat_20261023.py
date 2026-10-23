"""Repeat the unchanged full CPU admission in the exact replacement container."""
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')

if __name__ == '__main__':
    text = (PREP/'probe_sft_moment_direct_zero_runtime_20261021.py').read_text()
    before = 'sft-moment-direct-zero-approved-20261021.json'
    if text.count(before) != 1:
        raise ValueError('Exact replacement approval path required')
    text = text.replace(before,'sft-direct-zero-approved-20261023-02.json')
    exec(compile(text,'<unchanged-live-CPU-admission>','exec'),{'__name__':'__main__','__file__':__file__})
