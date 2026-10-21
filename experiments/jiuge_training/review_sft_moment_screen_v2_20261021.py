"""Pre-training hardware amendment; all six runs use the same new card pair."""
from pathlib import Path
from continue_sft_moment_screen_20261021 import moment_gate
from review_sft_moment_screen_20261021 import derive

BASE=Path(__file__).resolve().parent


def generated():
    text=derive((BASE/'review_sft_numa_v2_20261021.py').read_bytes())
    for before,after in (
        ('sft_moment_screen_20261021.contract.json','sft_moment_screen_v2_20261021.contract.json'),
        ('sft-moment-screen-source-manifest-20261021.json','sft-moment-screen-source-manifest-v2-20261021.json'),
        ('sft-moment-screen-independent-review-20261021.json','sft-moment-screen-independent-review-v2-20261021.json')):
        if text.count(before)!=1:raise ValueError('Exact hardware amendment review derivation required')
        text=text.replace(before,after)
    return text


def main():
    exec(compile(generated(),str(BASE/'review_sft_moment_screen_v2_20261021.generated.py'),'exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})


if __name__=='__main__':main()
