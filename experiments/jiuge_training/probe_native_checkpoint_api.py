"""CPU-only DCP data-API preflight with an explicit CPU CUDA-barrier shim.

The native writer unconditionally synchronizes CUDA even for all-CPU inputs.
The guarded no-op below is confined to this CPU preflight; the full7B run uses
the actual native CUDA synchronization and checkpoint path.
"""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import traceback

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    import numpy as np
    import torch
    torch.set_num_threads(1)
    rank = int(os.environ['RANK'])
    root = PREP / 'checkpoint-api-preflight-20261015-02'
    if rank == 0:
        root.mkdir(exist_ok=False)
        (root / 'checkpoint').mkdir()
    torch.distributed.init_process_group('gloo', timeout=timedelta(seconds=120))
    torch.distributed.barrier()
    receipt = {'schema': 'jiuge-native-dcp-cpu-preflight-v1', 'rank': rank, 'passed': False,
               'utc': datetime.now(timezone.utc).isoformat(), 'torch_version': str(torch.__version__),
               'scope': 'Small CPU AdamW/DCP tensor and RNG/common data APIs, with guarded CPU-only CUDA sync no-op; not a full7B or native CUDA path test',
               'full_7b_resume_verified': False}
    try:
        sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedObject
        torch.manual_seed(63100 + rank)
        np.random.seed(63100 + rank)
        random.seed(63100 + rank)
        master = torch.nn.Parameter(torch.arange(rank * 8, (rank + 1) * 8).float() / 16)
        optimizer = torch.optim.AdamW([master], lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.1)
        for step in (1, 2):
            master.grad = torch.sin(torch.arange(8).float() + step + rank)
            optimizer.step()
        values = {'parameter': master.detach().bfloat16().clone(), 'master': master.detach().clone(),
                  'exp_avg': optimizer.state[master]['exp_avg'].clone(),
                  'exp_avg_sq': optimizer.state[master]['exp_avg_sq'].clone()}
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch_cpu': torch.get_rng_state(), 'optimizer_step': optimizer.state[master]['step'].clone()}
        common = {'iteration': 2, 'scheduler': {'num_steps': 2, 'max_lr': 1e-5}}
        source = {key: ShardedTensor.from_rank_offsets(key, value, (0, rank, 2)) for key, value in values.items()}
        source.update(rng=ShardedObject('rng', rng, (2,), (rank,)), common=common)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(value.device.type != 'cpu' for value in values.values()):
            raise ValueError('CPU preflight must expose no GPUs and use only CPU tensors')
        receipt.update(cpu_only_cuda_barrier_shim=True, cuda_barrier_noop_calls=0)
        def cpu_only_barrier(*args, **kwargs):
            if torch.cuda.is_initialized():
                raise ValueError('No-op barrier may not be used with an initialized CUDA context')
            receipt['cuda_barrier_noop_calls'] += 1
        torch.cuda.synchronize = cpu_only_barrier
        dist_checkpointing.save(source, root / 'checkpoint')
        guidance = {key: ShardedTensor.from_rank_offsets(key, torch.empty_like(value), (0, rank, 2))
                    for key, value in values.items()}
        guidance.update(rng=ShardedObject('rng', None, (2,), (rank,)), common=common)
        loaded = dist_checkpointing.load(guidance, root / 'checkpoint')
        for key, value in values.items():
            if value.dtype != loaded[key].dtype or not torch.equal(value, loaded[key]):
                raise ValueError('Native DCP tensor did not restore exactly: ' + key)
        if loaded['common'] != common or loaded['rng']['python'] != rng['python']:
            raise ValueError('Native DCP common/Python state differs')
        if not torch.equal(loaded['rng']['torch_cpu'], rng['torch_cpu']) or not torch.equal(
                loaded['rng']['optimizer_step'], rng['optimizer_step']):
            raise ValueError('Native DCP CPU RNG/optimizer step differs')
        for old, new in zip(rng['numpy'], loaded['rng']['numpy']):
            if not np.array_equal(old, new):
                raise ValueError('Native DCP NumPy RNG differs')
        if torch.cuda.is_initialized():
            raise ValueError('CPU preflight unexpectedly initialized CUDA')
        receipt.update(passed=True, cuda_initialized=False, exact_tensor_fields=list(values),
                       sharded_rng_and_common_objects_exact=True,
                       source_sha256={name: hashlib.sha256((PREP / 'megatron-native-5be9626' / name).read_bytes()).hexdigest()
                                      for name in ('megatron/core/dist_checkpointing/serialization.py',
                                                   'megatron/core/dist_checkpointing/mapping.py')})
    except BaseException as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        with (root / f'rank-{rank}-receipt.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
        print(json.dumps(receipt), flush=True)
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
