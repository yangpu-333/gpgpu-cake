"""Independent review of all fresh compatibility cases and all112 frozen sources."""
from pathlib import Path
from review_sft_moment_direct_zero_20261021 import generated as previous
from continue_sft_moment_direct_zero_20261021 import moment_gate
BASE=Path(__file__).resolve().parent

def generated():
    text=previous()
    changes=[('sft-moment-direct-zero-driver-20261021-01','sft-direct-zero-driver-20261023-02'),
             ('sft-moment-direct-zero-screen-20261021-01','sft-direct-zero-screen-20261023-02'),
             ('sft-moment-direct-zero-source-manifest-v3-20261021.json','sft-direct-zero-source-manifest-20261023-02.json'),
             ('sft-moment-direct-zero-independent-review-20261021.json','sft-direct-zero-independent-review-20261023-02.json')]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact independent review paths required: '+before)
        text=text.replace(before,after)
    return text

if __name__=='__main__':
    exec(compile(generated(),'<optional-peak-independent-review>','exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})
