"""Keep the exact native SFT launcher; version only candidate entry and sources."""
from pathlib import Path
from run_sft_moment_case_20261021 import derive as previous

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source):
    text=previous(source)
    changes=[('sft_moment_entry_20261021.py','sft_moment_direct_zero_entry_20261021.py'),
             ('sft_local_moments_20261021.py','sft_local_moments_v3_20261021.py'),
             ("'run_sft_moment_case_20261021.py'", "'run_sft_moment_direct_zero_case_20261021.py','sft_moment_memory_guard_20261021.py'")]
    for before,after in changes:
        # The old entry occurs once in the generated command and once in its source snapshot.
        expected=2 if before=='sft_moment_entry_20261021.py' else 1
        if text.count(before)!=expected:raise ValueError('Exact launch derivation differs: '+before)
        text=text.replace(before,after)
    return text


if __name__=='__main__':
    text=derive((PREP/'run_sft_trace_case_20261020.py').read_bytes())
    exec(compile(text,'<direct-zero-native-SFT-launcher>','exec'),{'__name__':'__main__','__file__':__file__})
