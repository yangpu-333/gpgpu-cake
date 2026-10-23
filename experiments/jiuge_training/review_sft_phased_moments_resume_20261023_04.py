"""Review the same frozen scientific candidate under a separately authorized window."""
from review_sft_phased_moments_20261023 import generated as previous, moment_gate, phase_gate


def generated():
    text = previous()
    changes = (
        ('sft-phased-moments-source-manifest-20261023-03.json', 'sft-phased-moments-resume-source-manifest-20261023-04.json'),
        ('sft_phased_moments_20261023.contract.json', 'sft_phased_moments_resume_20261023_04.contract.json'),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact budget-only review adaptation required: '+before)
        text = text.replace(before, after)
    return text


if __name__ == '__main__':
    exec(compile(generated(), '<phased-budget-resume-independent-review>', 'exec'),
         {'__name__': '__main__', '__file__': __file__, 'moment_gate': moment_gate, 'phase_gate': phase_gate})
