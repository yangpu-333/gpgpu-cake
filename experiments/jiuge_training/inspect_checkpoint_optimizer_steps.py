"""CPU-only inspection of saved scalar optimizer step metadata, not tensor shards."""
import argparse
import io
import json
import os
from pathlib import Path
import sys


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path('/home/huids25/kda-jiuge-prep-20261014')
    sys.path.insert(0, str(root / 'megatron-native-5be9626'))
    import torch
    import torch.distributed.checkpoint as dcp
    torch.serialization.add_safe_globals([argparse.Namespace, torch.optim.AdamW, torch.optim.Adam])
    from checkpoint_load_compat import install
    install()
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict
    checkpoint = root / 'pinning-semantics-20261015-02/checkpoints/native-step2/iter_0000002'
    def steps(value, path='root'):
        result = []
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key) == 'step':
                    result.append({'path': path + '/step', 'value': float(item.item()) if torch.is_tensor(item) else item})
                elif isinstance(item, (dict, list, tuple)):
                    result.extend(steps(item, path + '/' + str(key)))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                result.extend(steps(item, path + '/' + str(index)))
        return result
    common = load_common_state_dict(checkpoint)
    result = {'iteration': common['iteration'], 'common_optimizer_steps': steps(common.get('optimizer')),
              'byte_objects': [], 'cuda_initialized': False}
    metadata = dcp.FileSystemReader(checkpoint).read_metadata()
    for key, row in metadata.state_dict_metadata.items():
        if key.startswith('optimizer') and type(row).__name__ == 'BytesStorageMetadata':
            target = {key: io.BytesIO()}
            dcp.load_state_dict(target, dcp.FileSystemReader(checkpoint), no_dist=True)
            value = target[key]
            if isinstance(value, io.BytesIO):
                value.seek(0)
                value = torch.load(value, map_location='cpu', weights_only=True)
            result['byte_objects'].append({'key': key, 'steps': steps(value)})
    result['cuda_initialized'] = torch.cuda.is_initialized()
    with (root / 'checkpoint-optimizer-step-inspection-20261016.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
