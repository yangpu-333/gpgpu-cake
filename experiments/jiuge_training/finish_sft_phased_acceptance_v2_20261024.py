"""One independent CPU reviewer for the new immutable V2 key package."""
from pathlib import Path


def main():
    base = Path(__file__).resolve().parent
    text = (base/'finish_sft_phased_acceptance_20261024.py').read_text(encoding='utf-8')
    changes = {
        'sft-phased-acceptance-postprocess-20261024-01':'sft-phased-acceptance-postprocess-20261024-02',
        'sft-phased-acceptance-driver-20261024-01':'sft-phased-acceptance-driver-20261024-02',
        'review_sft_phased_acceptance_20261024.py':'review_sft_phased_acceptance_v2_20261024.py',
        'sft-phased-acceptance-independent-review-20261024-01.json':'sft-phased-acceptance-independent-review-20261024-02.json',
    }
    for old,new in changes.items():
        assert text.count(old)==1, old
        text = text.replace(old,new)
    exec(compile(text,'<CPU-acceptance-postprocess-v2>','exec'),{'__name__':'__main__','__file__':__file__})


if __name__=='__main__':
    main()
