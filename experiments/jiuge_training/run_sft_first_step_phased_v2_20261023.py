"""Preserve failed first attempt; use only an optional-peak compatible entry."""
from pathlib import Path
import run_sft_direct_zero_first_step_20261023 as original

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
previous = original.derive


def derive(source, timeout_seconds=1800):
    text = previous(source,timeout_seconds)
    before = "str(PREP/'sft_direct_zero_first_step_entry_20261023.py')"
    if text.count(before) != 1:
        raise ValueError('Exact first-update entry substitution required')
    return text.replace(before,"str(PREP/'sft_phased_moment_entry_v2_20261023.py')")


if __name__ == '__main__':
    original.derive = derive
    original.APPROVAL = PREP/'sft-phased-moments-approved-20261023-07.json'
    original.main()
