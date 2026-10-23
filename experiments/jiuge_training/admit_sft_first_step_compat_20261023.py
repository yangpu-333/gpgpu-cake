"""Version orchestration paths after a pre-training observer failure."""
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source):
    text = source.decode()
    changes = (
        ('sft-direct-zero-first-step-20261023-01','sft-direct-zero-first-step-20261023-02',2),
        ('sft-direct-zero-first-step-approved-20261023.json','sft-direct-zero-approved-20261023-02.json',1),
        ('sft-moment-direct-zero-approved-20261021.json','sft-direct-zero-approved-20261023-02.json',1),
        ('run_sft_direct_zero_first_step_20261023.py','run_sft_first_step_compat_20261023.py',1),
    )
    for before, after, count in changes:
        if text.count(before) != count:
            raise ValueError('Exact standalone admission adaptation required: '+before)
        text = text.replace(before,after)
    return text


if __name__ == '__main__':
    exec(compile(derive((PREP/'admit_sft_direct_zero_first_step_20261023.py').read_bytes()),
                 '<optional-peak-one-update-admission>','exec'),{'__name__':'__main__','__file__':__file__})
