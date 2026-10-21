"""CPU-only check of DataLoader iterator startup RNG in the actual runtime."""
import hashlib
import json
from pathlib import Path


def main():
    import torch
    from torch.utils.data import DataLoader
    torch.set_num_threads(1)
    torch.manual_seed(63100)
    loader = DataLoader(list(range(6)), batch_size=1, num_workers=0, pin_memory=False)
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    saved = torch.get_rng_state().clone()
    continuation = next(iterator)
    continuous = torch.get_rng_state().clone()
    torch.set_rng_state(saved)
    restored = torch.get_rng_state().clone()
    iter(loader)
    native_restart = torch.get_rng_state().clone()
    torch.set_rng_state(saved)
    iter(loader)
    candidate_restart = torch.get_rng_state().clone()
    digest = lambda value: hashlib.sha256(value.numpy().tobytes()).hexdigest()
    result = {'schema': 'native-dataloader-startup-cpu-rng-probe-v1', 'torch_version': torch.__version__,
              'native_loader_source': 'megatron/training/datasets/data_samplers.py',
              'unchanged_continuous_iteration_keeps_rng': torch.equal(saved, continuous),
              'checkpoint_rng_restoration_exact': torch.equal(saved, restored),
              'iterator_recreation_advances_cpu_rng': not torch.equal(restored, native_restart),
              'two_same_checkpoint_restarts_match': torch.equal(native_restart, candidate_restart),
              'saved_rng_sha256': digest(saved), 'restarted_rng_sha256': digest(native_restart),
              'cuda_initialized': torch.cuda.is_initialized(),
              'scope': 'DataLoader iterator RNG only; not a full Megatron checkpoint test',
              'pin_memory_disabled_to_avoid_GPU_activity': True}
    assert all(result[key] for key in ('unchanged_continuous_iteration_keeps_rng', 'checkpoint_rng_restoration_exact',
                                      'iterator_recreation_advances_cpu_rng', 'two_same_checkpoint_restarts_match'))
    assert not result['cuda_initialized']
    output = Path('/home/huids25/kda-jiuge-prep-20261014/dataloader-rng-probe-20261015.json')
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
