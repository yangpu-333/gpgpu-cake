"""Independent review of the fresh phased setup comparison only."""
from pathlib import Path
import json
from review_sft_direct_zero_compat_20261023 import generated as previous
from continue_sft_moment_direct_zero_20261021 import moment_gate
BASE=Path(__file__).resolve().parent

def generated():
    text=previous()
    for before,after in [('sft-direct-zero-driver-20261023-02','sft-phased-moments-driver-20261023-07'),('sft-direct-zero-screen-20261023-02','sft-phased-moments-screen-20261023-07'),('sft-direct-zero-source-manifest-20261023-02.json','sft-phased-moments-source-manifest-20261023-07.json'),('sft-direct-zero-independent-review-20261023-02.json','sft-phased-moments-independent-review-20261023-07.json'),('sft_moment_direct_zero_v3_20261021.contract.json','sft_phased_moments_v2_20261023.contract.json')]:
        if text.count(before)!=1:raise ValueError('Exact phased review program adaptation required: '+before)
        text=text.replace(before,after)
    before="            binding = read(root/f'sft-numa-binding-rank-{rank}.json')"
    if text.count(before)!=1:raise ValueError('Exact phase provenance gate insertion required')
    text=text.replace(before,before+'\n            phase_gate(root,rank,item,plan)')
    return text


def phase_gate(root,rank,item,plan):
    setup=json.loads((root/f'sft-numa-setup-binding-rank-{rank}.json').read_text())
    update=json.loads((root/f'sft-numa-binding-rank-{rank}.json').read_text())
    expected=plan['arms'][item['policy']][rank]
    if update['allowed_after']!=expected or not all(v==expected for v in update['thread_affinity_after'].values()):
        raise ValueError('Actual rank-thread update placement differs')
    if item['policy']=='baseline':
        if setup['allowed_after']!=plan['topology']['allowed_cpus']:
            raise ValueError('Baseline must remain unbound from setup')
    elif (len(setup['allowed_after'])!=8
          or setup['selected_cpu_nodes']==[setup['assigned_gpu']['node']]
          or update['selected_cpu_nodes']!=[update['assigned_gpu']['node']]):
        raise ValueError('Control/candidate setup and update phases were not observed')


if __name__=='__main__':
    exec(compile(generated(),'<phased-independent-review>','exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate,'phase_gate':phase_gate})
