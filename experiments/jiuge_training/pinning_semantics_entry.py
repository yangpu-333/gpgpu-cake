"""Untimed real-model inputs, per-token loss, scheduler and RNG observations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--semantics-root', type=Path, required=True)
    parser.add_argument('--resume-audit-campaign', type=Path)
    args, remaining = parser.parse_known_args()
    sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
    import numpy as np
    import torch
    import megatron.training.training as training
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    from input_evidence import install_input_observer
    rank = int(os.environ['RANK'])
    folder = args.semantics_root / f'rank-{rank}'
    folder.mkdir(parents=True, exist_ok=False)
    state = {'step': 0, 'completed_steps': [], 'loss_calls': [], 'inputs': [], 'scheduler': None,
             'rows': [], 'run_returned_successfully': False}
    resume = None
    if args.resume_audit_campaign:
        from pinning_resume_audit import ResumeAudit
        resume = ResumeAudit(args.resume_audit_campaign, rank)
    setup = training.setup_model_and_optimizer
    def with_scheduler(*values, **keywords):
        result = setup(*values, **keywords)
        state['scheduler'] = result[2]
        from megatron.training import get_args
        state['step'] = int(get_args().iteration)
        state['initial_iteration'] = state['step']
        state['initial_consumed_train_samples'] = int(get_args().consumed_train_samples)
        # Observe restoration before native DataLoader iterator construction.
        save_raw('after-setup-rng.pt', rng())
        save_raw('after-setup-scheduler.pt', state['scheduler'].state_dict())
        if resume:
            resume.after_setup(result[1], state['step'])
        return result
    training.setup_model_and_optimizer = with_scheduler

    def save_raw(name, values):
        path = folder / name
        torch.save(values, path)
        state['rows'].append({'file': name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})

    def rng():
        numpy_state = np.random.get_state()
        return {'python': random.getstate(),
                'numpy': [numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]],
                'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state(),
                'model_parallel_cuda': {k: v.cpu().clone() for k, v in get_cuda_rng_tracker().get_states().items()}}

    schedule = training.get_forward_backward_func
    def get_schedule(*values, **keywords):
        original = schedule(*values, **keywords)
        def observed(*items, **options):
            callback = options.get('forward_step_func') or (items[0] if items else None)
            install_input_observer(torch, callback, folder, state['inputs'])
            namespace = callback.__globals__
            loss_func = namespace.get('loss_func')
            if not callable(loss_func):
                raise ValueError('Pinned native loss callback unavailable')
            if not getattr(loss_func, '_jiuge_loss_observer', False):
                def captured(loss_mask, output_tensor, *extra, **kw):
                    index = len(state['loss_calls']) + 1
                    raw = {'per_token_loss': output_tensor.detach().cpu().clone(),
                           'loss_mask': loss_mask.detach().cpu().clone(), 'native_step_call': state['step']}
                    result = loss_func(loss_mask, output_tensor, *extra, **kw)
                    raw['reduced_loss'] = result[0].detach().cpu().clone()
                    raw['num_tokens'] = result[1].detach().cpu().clone()
                    save_raw(f'loss-call-{index}.pt', raw)
                    state['loss_calls'].append({'call': index, 'step': state['step']})
                    return result
                captured._jiuge_loss_observer = True
                namespace['loss_func'] = captured
            return original(*items, **options)
        return observed
    training.get_forward_backward_func = get_schedule
    step = training.train_step
    def observed_step(*values, **keywords):
        state['step'] += 1
        index = state['step']
        save_raw(f'before-step-{index}-rng.pt', rng())
        result = step(*values, **keywords)
        save_raw(f'after-step-{index}-rng.pt', rng())
        scheduler = state['scheduler']
        save_raw(f'after-step-{index}-scheduler.pt', scheduler.state_dict() if scheduler is not None else {})
        state['completed_steps'].append(index)
        if resume:
            resume.record_loss(result)
        return result
    training.train_step = observed_step
    sys.argv = [str(PREP / 'pinning_entry.py'), *remaining]
    try:
        runpy.run_path(str(PREP / 'pinning_entry.py'), run_name='__main__')
        state['run_returned_successfully'] = True
    except SystemExit as exc:
        state['run_returned_successfully'] = exc.code in (None, 0)
        raise
    finally:
        if resume:
            resume.finish(state['run_returned_successfully'], state['completed_steps'])
        receipt = {'schema': 'jiuge7b-pinning-semantics-observation-v1', 'rank': rank,
                   'completed_step_calls': len(state['completed_steps']),
                   'completed_native_steps': state['completed_steps'],
                   'initial_iteration': state.get('initial_iteration'),
                   'initial_consumed_train_samples': state.get('initial_consumed_train_samples'),
                   'run_returned_successfully': state['run_returned_successfully'],
                   'loss_calls': state['loss_calls'],
                   'actual_inputs': state['inputs'], 'files': state['rows'],
                   'formal_performance_passed': False}
        with (folder / 'semantics.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)


if __name__ == '__main__':
    main()
