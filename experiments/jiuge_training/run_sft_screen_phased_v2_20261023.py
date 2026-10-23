"""Unchanged35-step screen launcher with a versioned telemetry-only entry."""
from pathlib import Path
from run_sft_moment_direct_zero_case_20261021 import derive as previous

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source):
    text = previous(source)
    before = "str(PREP/'sft_moment_direct_zero_entry_20261021.py')"
    if text.count(before) != 1:
        raise ValueError('Exact screen entry substitution required')
    return text.replace(before,"str(PREP/'sft_phased_moment_entry_v2_20261023.py')")


if __name__ == '__main__':
    exec(compile(derive((PREP/'run_sft_trace_case_20261020.py').read_bytes()),
                 '<optional-peak-screen-launcher>','exec'),{'__name__':'__main__','__file__':__file__})
