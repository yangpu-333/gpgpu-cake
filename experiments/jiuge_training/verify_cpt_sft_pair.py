"""Compare real CPT/SFT admission inputs on the new dual-GPU allocation."""
import hashlib
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    import torch
    folders = [PREP/'runs'/name for name in ('cpt-native-admission-20261017-01',
                                             'sft-native-admission-20261017-01')]
    def native_arguments(folder):
        values=json.loads((folder/'rank-0/entry.json').read_text())['argv']
        position=values.index('--data-path')
        return [*values[:position+1], '<TASK_DATA_PREFIX>', *values[position+2:]]
    if native_arguments(folders[0]) != native_arguments(folders[1]):
        raise ValueError('Actual native CPT/SFT arguments differ beyond the task data prefix')
    results = []
    for rank in range(2):
        for batch in range(1,4):
            raw = [folder/f'rank-{rank}/actual-input-{batch}.pt' for folder in folders]
            first, second = [torch.load(path,map_location='cpu',weights_only=True) for path in raw]
            fields = ('tokens','labels','position_ids','attention_mask')
            equal = {key:torch.equal(first[key],second[key]) for key in fields}
            if not all(equal.values()) or not (first['loss_mask']==1).all().item():
                raise ValueError('CPT/SFT inputs or original full-text loss mask differ')
            mask = second['loss_mask']
            if not ((mask==0)|(mask==1)).all().item() or not 0<mask.sum().item()<mask.numel():
                raise ValueError('Distinct real answer supervision required')
            results.append({'rank':rank,'batch':batch,'identical_fields':equal,
                            'CPT_supervised_targets':first['loss_mask'].numel(),
                            'SFT_supervised_targets':int(mask.sum()),
                            'raw_sha256':[hashlib.sha256(path.read_bytes()).hexdigest() for path in raw]})
    result = {'schema':'jiuge7b-cpt-sft-real-input-pair-v1','passed':True,'batches':results,
              'same_real_input_token_stream_labels_attention_and_positions':True,
              'only_supervised_loss_mask_differs':True,'cuda_initialized':torch.cuda.is_initialized(),
              'native_model_optimizer_training_arguments_equal_except_data_prefix':True,
              'formal_performance_comparison':False}
    path = PREP/'sft-tracing-20261017-01/cpt-sft-input-pair-verification.json'
    with path.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
