"""Three-step full-state diagnostic of the actual frozen HDO and isolated candidate.

This small optimizer diagnostic is not the full 7B model acceptance gate.
"""
import argparse
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
from pinning_candidate import load_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output-name', default='optimizer-diagnostic')
    parser.add_argument('--all-bf16', action='store_true')
    parser.add_argument('--fractions', nargs='+', type=float, default=[1.0])
    args = parser.parse_args()
    out = args.campaign / args.output_name
    out.mkdir(exist_ok=False)
    lock = Path('/home/huids25/kda-jiuge-prep-20261014/two-gpu-experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                            '--format=csv,noheader'], capture_output=True, text=True, check=True)
    if query.stdout.strip():
        raise RuntimeError('GPU compute already active')
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(63100)
    native = load_class(args.campaign / 'native_hybrid_optimizer.py', 'jiuge_native_hdo')
    candidate = load_class(args.campaign / 'candidate_hybrid_optimizer.py', 'jiuge_candidate_hdo')
    bases = [torch.randn(17, 32).bfloat16(), torch.randn(2048).bfloat16(), torch.randn(3, 5)]
    if args.all_bf16:
        bases = [p.bfloat16() for p in bases]
    stimuli = [[torch.randn(p.shape) for p in bases] for _ in range(3)]
    raw = {}
    checkpoints = []

    def snapshot(opt, params, loss, gradients):
        torch.cuda.synchronize()
        states = []
        for p in params:
            item = opt.state[p]
            states.append({k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                           for k, v in item.items()})
        return {'loss': loss.detach().cpu().clone(), 'gradients': gradients,
                'parameters': [p.detach().cpu().clone() for p in params], 'states': states}

    def update(opt, params, step):
        opt.zero_grad()
        loss = sum(((p.float() * x.cuda()).square().mean()) for p, x in zip(params, stimuli[step]))
        loss.backward()
        gradients = [p.grad.detach().cpu().clone() for p in params]
        opt.step()
        return snapshot(opt, params, loss, gradients)

    for fraction in args.fractions:
        arms = {}
        for arm in ('native', 'control', 'optimized'):
            cls = candidate if arm == 'optimized' else native
            params = [torch.nn.Parameter(p.cuda().clone()) for p in bases]
            opt = cls(params, offload_fraction=fraction, cpu_optimizer_cls=torch.optim.AdamW,
                      gpu_optimizer_cls=torch.optim.AdamW, param_update_in_fp32=True,
                      pin_cpu_params=True, pin_cpu_grads=True, overlap_cpu_optimizer_d2h_h2d=True,
                      lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
            masters = list(opt.gpu_params_map_cpu_copy.values())
            pinning = [p.is_pinned() for p in masters]
            steps = []
            for step in range(3):
                steps.append(update(opt, params, step))
                if step == 1:
                    checkpoint = {'parameters': [p.detach().cpu().clone() for p in params],
                                  'optimizer': copy.deepcopy(opt.state_dict())}
            if arm == 'optimized' and not all(pinning):
                raise AssertionError('Candidate CPU master is not pinned')
            arms[arm] = {'cpu_master_pinned': pinning, 'steps': steps}
            checkpoint_path = out / f'{fraction}-{arm}-checkpoint.pt'
            torch.save(checkpoint, checkpoint_path)
            # Only the known torch optimizer class appears in our own state dict.
            with torch.serialization.safe_globals([torch.optim.AdamW]):
                restored = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            restore_params = [torch.nn.Parameter(p.cuda().clone()) for p in restored['parameters']]
            restore = cls(restore_params, offload_fraction=fraction, cpu_optimizer_cls=torch.optim.AdamW,
                          gpu_optimizer_cls=torch.optim.AdamW, param_update_in_fp32=True,
                          pin_cpu_params=True, pin_cpu_grads=True, overlap_cpu_optimizer_d2h_h2d=True,
                          lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
            restore.load_state_dict(restored['optimizer'])
            arms[arm]['resumed_step3'] = update(restore, restore_params, 2)
            checkpoints.append(str(checkpoint_path.name))
            del opt, params, restore, restore_params
            torch.cuda.empty_cache()
        raw[str(fraction)] = arms
    torch.save(raw, out / 'all-three-step-states.pt')
    rows = []

    def compare(a, b, name):
        if torch.is_tensor(a):
            x, y = a.double(), b.double()
            ok = bool(torch.allclose(x, y, atol=3e-4, rtol=1e-3, equal_nan=False))
            rows.append({'field': name, 'numel': a.numel(), 'passed': ok,
                         'max_abs': float((x-y).abs().max()) if a.numel() else 0.0})
        elif isinstance(a, dict):
            if set(a) != set(b):
                raise AssertionError('State keys differ')
            for key in a:
                compare(a[key], b[key], f'{name}/{key}')
        elif isinstance(a, list):
            if len(a) != len(b):
                raise AssertionError('List lengths differ')
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f'{name}/{i}')
        elif a != b:
            raise AssertionError(f'Scalar differs: {name}')
    for fraction, arms in raw.items():
        for arm in ('control', 'optimized'):
            compare(arms['native']['steps'], arms[arm]['steps'], f'{fraction}/{arm}')
        for arm, values in arms.items():
            compare(values['steps'][2], values['resumed_step3'], f'{fraction}/{arm}/resume')
    receipt = {'schema': 'jiuge-actual-hdo-small-diagnostic-v1',
               'scope': 'Actual HDO, small heterogeneous tensors; not full 7B',
               'offload_fractions': args.fractions, 'all_bf16': args.all_bf16,
               'comparison_dtype': 'CPU float64', 'atol': 3e-4, 'rtol': 1e-3,
               'rows': rows, 'passed': all(r['passed'] for r in rows),
               'pinning': {f: {a: v['cpu_master_pinned'] for a, v in arms.items()} for f, arms in raw.items()},
               'files_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('*.pt')},
               'full_model_correctness_passed': False, 'formal_performance_passed': False}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'rows'}), flush=True)
    if not receipt['passed']:
        raise AssertionError('Numerical diagnostic failed')


if __name__ == '__main__':
    main()
