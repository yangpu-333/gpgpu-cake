"""Six fresh cases after standalone memory admission; no old case reuse."""
from pathlib import Path
from continue_sft_moment_direct_zero_20261021 import derive as previous,moment_gate

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source):
    text = previous(source)
    changes = (
        ('sft-moment-direct-zero-screen-20261021-01','sft-direct-zero-screen-20261023-02'),
        ('sft-moment-direct-zero-approved-20261021.json','sft-direct-zero-approved-20261023-02.json'),
        ('run_sft_moment_direct_zero_case_20261021.py','run_sft_screen_compat_20261023.py'),
        ('sft-moment-direct-zero-20261021-','sft-direct-zero-20261023-02-'),
        ('sft-moment-direct-zero-key-results-20261021.zip','sft-direct-zero-key-results-20261023-02.zip'),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact fresh compatibility controller required: '+before)
        text = text.replace(before,after)
    return text


if __name__ == '__main__':
    exec(compile(derive((PREP/'continue_sft_numa_v2_20261021.py').read_bytes()),
                 '<optional-peak-six-fresh-cases>','exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})
