"""Independent CPU review of the fresh direct-zero six-case screen only."""
from pathlib import Path
from review_sft_moment_screen_v2_20261021 import generated as previous
from continue_sft_moment_direct_zero_20261021 import moment_gate

BASE=Path(__file__).resolve().parent


def generated():
    text=previous()
    changes=[('sft-moment-screen-driver-20261021-01','sft-moment-direct-zero-driver-20261021-01'),
             ('sft-moment-screen-20261021-01','sft-moment-direct-zero-screen-20261021-01'),
             ('sft-moment-screen-key-results.zip','sft-moment-direct-zero-key-results.zip'),
             ('sft_moment_screen_v2_20261021.contract.json','sft_moment_direct_zero_v3_20261021.contract.json'),
             ('sft-moment-screen-source-manifest-v2-20261021.json','sft-moment-direct-zero-source-manifest-v3-20261021.json'),
             ('sft-moment-screen-independent-review-v2-20261021.json','sft-moment-direct-zero-independent-review-20261021.json')]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact independent review adaptation required: '+before)
        text=text.replace(before,after)
    return text


if __name__=='__main__':
    exec(compile(generated(),'<fresh-direct-zero-independent-review>','exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})
