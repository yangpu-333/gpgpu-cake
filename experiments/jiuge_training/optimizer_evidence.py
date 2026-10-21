"""Summarize actual native AdamW state after training, outside trace windows."""
import json


def save_optimizer_evidence(optimizer,path):
    import torch
    pending=[optimizer]
    seen=set()
    leaves=[]
    while pending:
        current=pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        children=getattr(current,'chained_optimizers',None)
        if children:
            pending.extend(children)
        elif hasattr(current,'optimizer'):
            pending.append(current.optimizer)
        else:
            leaves.append(current)
    records=[]
    samples=[]
    for leaf in leaves:
        moments=masters=0
        steps=[]
        entries=0
        state=getattr(leaf,'state',{})
        for ordinal,(parameter,item) in enumerate(state.items()):
            if 'exp_avg' not in item or 'exp_avg_sq' not in item:
                continue
            entries+=1
            moments+=item['exp_avg'].numel()
            if 'master_param' in item:
                masters+=item['master_param'].numel()
            step=item.get('step')
            if step is not None:
                steps.append(float(step.item() if torch.is_tensor(step) else step))
            if ordinal<4:
                samples.append({key:value.detach().reshape(-1)[:16].cpu().clone()
                                for key,value in item.items() if torch.is_tensor(value)})
        records.append({'class':type(leaf).__module__+'.'+type(leaf).__name__,
                        'adam_state_entries':entries,'elements_with_both_adam_moments':moments,
                        'elements_with_master_param':masters,'min_step':min(steps) if steps else None,
                        'max_step':max(steps) if steps else None,
                        'offload_fraction':getattr(leaf,'offload_fraction',None)})
    torch.save(samples,path.with_suffix('.samples.pt'))
    result={'schema':'jiuge7b-actual-optimizer-evidence-v1','optimizers':records,
            'all_state_saved':False,'selected_raw_state_samples_saved':True}
    with path.open('x') as stream:
        json.dump(result,stream,indent=2)
    return result
