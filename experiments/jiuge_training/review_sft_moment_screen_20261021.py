"""Reuse the closed independent review with explicit new three-arm semantics."""
import hashlib
from pathlib import Path
from continue_sft_moment_screen_20261021 import moment_gate

BASE=Path(__file__).resolve().parent
EXPECTED='4c6a73253f831702851badd319583e384b1e5acd55ee48a262a429a6fdffaaff'


def derive(source):
    if hashlib.sha256(source).hexdigest()!=EXPECTED:raise ValueError('Frozen CPU review changed')
    text=source.decode()
    old="""    expected_order = contract['order_with_two_distinct_candidates' if 'socket-split8' in plan['policies']
                              else 'order_with_one_distinct_candidate']"""
    changes=[('sft-numa-driver-20261021-02','sft-moment-screen-driver-20261021-01'),
             ('sft-numa-diagnostic-20261021-02','sft-moment-screen-20261021-01'),
             ('sft-numa-key-results-v2.zip','sft-moment-screen-key-results.zip'),
             ('sft_numa_v2_20261021.contract.json','sft_moment_screen_20261021.contract.json'),
             (old,"    expected_order = contract['order']"),
             ('sft-numa-source-manifest-v2-20261021.json','sft-moment-screen-source-manifest-20261021.json'),
             ("plan['policies'][item['policy']][rank]","plan['arms'][item['policy']][rank]"),
             ("binding['mode'] != item['policy']","binding['mode'] != contract['arm_modes'][item['policy']]"),
             ("            for phase in ('after-first-update', 'after-training'):","            moment_gate(root,rank,item['policy'],binding)\n            for phase in ('after-first-update', 'after-training'):"),
             ("for policy in plan['policies']:","for policy in plan['arms']:"),
             ("if x['policy'] == 'unbound'","if x['policy'] == 'baseline'"),
             ("set(plan['policies'])-{'unbound'}","set(plan['arms'])-{'baseline'}"),
             ("    result = {'schema':", "    candidate=[x['diagnostic_slow_rank_mean_seconds_per_step'] for x in cases if x['policy']=='local-moments']\n    controls=[x['diagnostic_slow_rank_mean_seconds_per_step'] for x in cases if x['policy']=='affinity-control']\n    screens['local-moments'].update(both_candidate_runs_faster_than_both_affinity_control_runs=max(candidate)<min(controls),eligible_for_own_full_SFT_acceptance=max(candidate)<min(controls) and max(candidate)<min(baseline))\n    result = {'schema':"),
             ('jiuge7b-sft-numa-independent-diagnostic-review-v2','jiuge7b-sft-moment-screen-independent-review-v1'),
             ('fresh unbound SFT runs','fresh unbound SFT runs; affinity-only additional control'),
             ('sft-numa-independent-review-v2-20261021.json','sft-moment-screen-independent-review-20261021.json')]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact independent review derivation failed: '+before)
        text=text.replace(before,after)
    return text


def main():
    text=derive((BASE/'review_sft_numa_v2_20261021.py').read_bytes())
    exec(compile(text,str(BASE/'review_sft_moment_screen_20261021.generated.py'),'exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})


if __name__=='__main__':main()
