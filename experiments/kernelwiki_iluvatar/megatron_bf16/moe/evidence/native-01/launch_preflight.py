"""Untimed native MoE availability probe; no replacement expert/router operations."""
import argparse
import fcntl
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import traceback


p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--moe-receipt', type=Path, required=True)
p.add_argument('--delegate-launcher', type=Path, required=True)
a, rest = p.parse_known_args()
assert not a.moe_receipt.exists()
bf = a.delegate_launcher.parent
sys.path.insert(0, str(bf.parent / 'megatron_training_entry'))
import runtime_compat
original_prepare = runtime_compat.prepare_local_runtime
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = {'schema': 'bi-v150-native-moe-availability-v1', 'status': 'initializing',
          'wrapper_sha256': sha(__file__), 'moe_calls': [], 'expert_calls': [],
          'probability_gradients': [], 'training_steps': [], 'candidate_enabled': False,
          'performance_claims': [], 'scope': 'Only native single-device three-step MoE availability, '
          'actual backend/token counts/gradient/optimizer dtype observations; untimed scalar CPU copies. '
          'No candidate correctness comparison, GroupedGEMM, throughput, convergence or production claim.'}


def prepare(root):
    result = original_prepare(root)
    import torch
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.experts import SequentialMLP
    import megatron.training.training as training
    native_layer, native_experts, native_step = MoELayer.forward, SequentialMLP.forward, training.train_step
    report['native_source_sha256'] = {name: sha(Path(root) / name) for name in
        ('megatron/core/transformer/moe/moe_layer.py', 'megatron/core/transformer/moe/experts.py',
         'megatron/core/transformer/moe/router.py', 'megatron/core/transformer/moe/token_dispatcher.py',
         'megatron/core/transformer/mlp.py')}

    def layer(self, *args, **kwargs):
        cfg = self.config
        output = native_layer(self, *args, **kwargs)
        report['moe_calls'].append({'layer': self.layer_number,
            'experts': type(self.experts).__module__ + '.' + type(self.experts).__name__,
            'router': type(self.router).__module__ + '.' + type(self.router).__name__,
            'dispatcher': type(self.token_dispatcher).__module__ + '.' + type(self.token_dispatcher).__name__,
            'config': {name: getattr(cfg, name, None) for name in
                ('num_moe_experts', 'moe_router_topk', 'moe_grouped_gemm', 'moe_token_dispatcher_type',
                 'moe_apply_probs_on_input', 'moe_aux_loss_coeff', 'moe_z_loss_coeff',
                 'gated_linear_unit', 'bias_activation_fusion', 'masked_softmax_fusion')}})
        return output

    def experts(self, hidden, counts, probabilities):
        row = {'hidden_shape': list(hidden.shape), 'hidden_dtype': str(hidden.dtype),
               'tokens_per_expert': counts.detach().cpu().tolist(),
               'probability_shape': list(probabilities.shape), 'probability_dtype': str(probabilities.dtype),
               'probability_requires_grad': probabilities.requires_grad}
        report['expert_calls'].append(row)
        if probabilities.requires_grad:
            def observe_gradient(grad):
                report['probability_gradients'].append({'shape': list(grad.shape), 'dtype': str(grad.dtype),
                    'finite': bool(torch.isfinite(grad).all().item()),
                    'nonzero_elements': int(torch.count_nonzero(grad).item())})
                return grad
            probabilities.register_hook(observe_gradient)
        return native_experts(self, hidden, counts, probabilities)

    signature = inspect.signature(native_step)
    def step(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        output = native_step(*args, **kwargs)
        optimizer = bound.arguments['optimizer']
        chunks = bound.arguments['model']
        params = [param for chunk in chunks for param in chunk.parameters()]
        children = getattr(optimizer, 'chained_optimizers', [optimizer])
        masters = [param for child in children for group in getattr(child, 'fp32_from_float16_groups', [])
                   for param in group]
        grads = [param.main_grad for param in params if getattr(param, 'main_grad', None) is not None]
        report['training_steps'].append({'optimizer_classes': [type(child).__name__ for child in children],
            'native_optimizer_classes': [type(child.optimizer).__name__ for child in children],
            'parameter_tensors': len(params), 'parameters': sum(param.numel() for param in params),
            'parameter_dtypes': sorted({str(param.dtype) for param in params}),
            'main_grad_tensors': len(grads), 'main_grad_dtypes': sorted({str(grad.dtype) for grad in grads}),
            'main_grads_finite': all(bool(torch.isfinite(grad).all().item()) for grad in grads),
            'master_tensors': len(masters), 'master_dtypes': sorted({str(param.dtype) for param in masters}),
            'masters_finite': all(bool(torch.isfinite(param).all().item()) for param in masters)})
        return output

    MoELayer.forward, SequentialMLP.forward, training.train_step = layer, experts, step
    return result


runtime_compat.prepare_local_runtime = prepare
sys.argv = [str(a.delegate_launcher)] + rest
try:
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runpy.run_path(str(a.delegate_launcher), run_name='__main__')
    assert len(report['training_steps']) == 3 and len(report['moe_calls']) == 12
    assert len(report['expert_calls']) == len(report['probability_gradients']) == 12
    assert all(row['main_grads_finite'] and row['masters_finite'] for row in report['training_steps'])
    assert all(row['finite'] for row in report['probability_gradients'])
    assert all(sum(row['tokens_per_expert']) == 512 for row in report['expert_calls'])
    report['status'] = 'native_available'
except BaseException as error:
    report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    raise
finally:
    a.moe_receipt.parent.mkdir(parents=True, exist_ok=True)
    a.moe_receipt.write_text(json.dumps(report, indent=2) + '\n')
