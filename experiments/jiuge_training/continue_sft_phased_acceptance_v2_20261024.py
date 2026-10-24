"""Preserve two completed references; repeat candidate audit and every later gate."""
from sft_phased_acceptance_recovery_20261024 import configure, alias_completed


def generated():
    core=configure()
    text=(core.PREP/'continue_sft_phased_acceptance_20261024.py').read_text()
    changes=[
        ("'run_sft_phased_acceptance_20261024.py'","'run_sft_phased_acceptance_v2_20261024.py'"),
        ("        for arm in ('native','control','optimized'):","        alias_completed(numeric,semantics,state,cases)\n        for arm in ('optimized',):"),
        ("'phased-audit-'+arm+'-20261024-01'","'phased-audit-'+arm+'-20261024-02'"),
        ("phased-native-checkpoint-20261024-01","phased-native-checkpoint-20261024-02"),
        ("'phased-resume-'+arm+'-20261024-01'","'phased-resume-'+arm+'-20261024-02'"),
        ("'sft-phased-acceptance-correctness-20261024-01.zip'","'sft-phased-acceptance-correctness-20261024-02.zip'"),
        ("'sft-phased-acceptance-final-20261024-01.zip'","'sft-phased-acceptance-final-20261024-02.zip'"),
        ("'phased20261024'+round_name","'phased20261024v2'+round_name"),
    ]
    return core.replace_exact(text,changes)


if __name__=='__main__':
    exec(compile(generated(),'<same-acceptance-observer-recovery>','exec'),
         {'__name__':'__main__','__file__':__file__,'alias_completed':alias_completed})
