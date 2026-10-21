"""Read actual saved 7B common state using CPU only; no tensor shards loaded."""
import argparse
import json
import os
from pathlib import Path
import sys
from checkpoint_load_compat import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    root = Path('/home/huids25/kda-jiuge-prep-20261014')
    sys.path.insert(0, str(root / 'megatron-native-5be9626'))
    import torch
    import torch.distributed as dist
    torch.serialization.add_safe_globals([argparse.Namespace])
    from megatron.core.dist_checkpointing.serialization import load_common_state_dict
    evidence = install()
    initialized = False
    try:
        if 'RANK' in os.environ:
            dist.init_process_group('gloo')
            initialized = True
        value = load_common_state_dict(args.checkpoint)
        evidence.update(rank=int(os.environ.get('RANK', 0)),
                        actual_iteration=value.get('iteration'),
                        consumed_train_samples=getattr(value.get('args'), 'consumed_train_samples', None),
                        common_keys=sorted(value),
                        cuda_initialized=torch.cuda.is_initialized(),
                        initialized_gloo=initialized,
                        passed=value.get('iteration') == 2 and not torch.cuda.is_initialized())
        output = Path(str(args.output).replace('{rank}', str(evidence['rank'])))
        with output.open('x') as stream:
            json.dump(evidence, stream, indent=2)
        print(json.dumps(evidence), flush=True)
        if not evidence['passed']:
            raise AssertionError('Actual local common-state restore did not pass')
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
