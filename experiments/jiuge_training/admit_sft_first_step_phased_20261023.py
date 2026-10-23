"""Version orchestration paths after a pre-training observer failure."""
from pathlib import Path
import json
import math
import re
import socket

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source):
    text = source.decode()
    changes = (
        ('sft-direct-zero-first-step-20261023-01','sft-phased-first-step-20261023-03',2),
        ('sft-direct-zero-first-step-approved-20261023.json','sft-phased-moments-approved-20261023-03.json',1),
        ('sft-moment-direct-zero-approved-20261021.json','sft-phased-moments-approved-20261023-03.json',1),
        ('run_sft_direct_zero_first_step_20261023.py','run_sft_first_step_phased_20261023.py',1),
    )
    for before, after, count in changes:
        if text.count(before) != count:
            raise ValueError('Exact standalone admission adaptation required: '+before)
        text = text.replace(before,after)
    before = "        for rank in (0,1):"
    if text.count(before) != 1:
        raise ValueError('Exact native first-update verification insertion required')
    text=text.replace(before,'        native_gate(folder)\n'+before)
    return text


def native_gate(folder):
    for rank in (0,1):
        entry=json.loads((folder/f'rank-{rank}/entry.json').read_text())
        weights=entry['weight_loading']
        optimizers=entry['actual_optimizer_state']['optimizers']
        if (entry['status']!='completed' or entry['native_files_checked']!=600
                or entry['completed_native_steps']!=[{'step':1,'skipped':0}]
                or weights['source_tensor_count']!=339 or weights['source_parameter_elements']!=7668831744
                or not weights['all_loaded_shard_bytes_verified']
                or sum(x['adam_state_entries'] for x in optimizers)!=199
                or any(x['min_step']!=1 or x['max_step']!=1 for x in optimizers if x['adam_state_entries'])):
            raise ValueError('Actual complete native first update was not verified')
    losses=[float(x) for x in re.findall(r'lm loss:\s*([\d.E+\-]+)',(folder/'stdout.log').read_text())]
    if len(losses)!=1 or not math.isfinite(losses[0]):
        raise ValueError('Exactly one finite native SFT loss required')
    with (folder/'native-first-step-verification.json').open('x') as stream:
        json.dump({'passed':True,'updates_each_rank':1,'native_files_each_rank':600,
                   'source_tensors':339,'AdamW_groups_each_rank':199,'native_loss':losses[0],
                   'full_state_cross_run_acceptance':False,'formal_performance_evidence':False},stream,indent=2)


if __name__ == '__main__':
    live=PREP/('sft-moment-direct-zero-runtime-'+socket.gethostname())/'phased-admission.json'
    result=json.loads(live.read_text())
    if not result['passed'] or result['GPU_used']:
        raise ValueError('Actual threaded CPU phase/state/resume admission must pass first')
    exec(compile(derive((PREP/'admit_sft_direct_zero_first_step_20261023.py').read_bytes()),
                 '<phased-one-update-admission>','exec'),{'__name__':'__main__','__file__':__file__,'native_gate':native_gate})
