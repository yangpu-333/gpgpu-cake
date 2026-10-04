"""Process-local activation hook; native MLP/GEMMs and training stay intact."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--swiglu-candidate', type=Path)
p.add_argument('--swiglu-receipt', type=Path, required=True)
p.add_argument('--capture-swiglu', type=Path)
p.add_argument('--gradnorm-launcher', type=Path, required=True)
a, rest = p.parse_known_args()
assert not a.swiglu_receipt.exists()
bf = Path(rest[rest.index('--bf16-root') + 1])
sys.path.insert(0, str(bf.parent / 'megatron_training_entry'))
import runtime_compat
original_prepare = runtime_compat.prepare_local_runtime
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {'enabled': a.swiglu_candidate is not None, 'calls': 0,
           'native_calls': 0, 'fallback_calls': 0, 'shapes': {},
           'wrapper_sha256': sha(Path(__file__)),
           'candidate_sha256': sha(a.swiglu_candidate) if a.swiglu_candidate else None}


def prepare(root):
    result = original_prepare(root)
    import torch
    import megatron.core.transformer.mlp as mlp
    import megatron.core.fusions.fused_bias_swiglu as fused
    receipt['native_source_sha256'] = {
        name: sha(Path(root) / name) for name in
        ('megatron/core/transformer/mlp.py',
         'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    receipt['native_jit_callable'] = str(type(fused.swiglu))
    receipt['native_compiled_callable'] = hasattr(fused.swiglu, '_torchdynamo_orig_callable')
    native = mlp.bias_swiglu_impl
    candidate = None
    if a.swiglu_candidate:
        spec = importlib.util.spec_from_file_location('swiglu_candidate', a.swiglu_candidate)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        if hasattr(candidate, '_ext'):
            artifact = Path(candidate._ext.__file__).resolve()
            receipt['extension'] = {'path': str(artifact), 'sha256': sha(artifact)}

    # Separate untimed observation verifies actual model configuration, not flags alone.
    original_forward = mlp.MLP.forward

    observed_configs = set()

    def observed_forward(self, hidden_states, per_token_scale=None, **kwargs):
        cfg = self.config
        if id(cfg) not in observed_configs:
            recorded = {'gated_linear_unit': cfg.gated_linear_unit,
                        'bias_activation_fusion': cfg.bias_activation_fusion,
                        'activation_func': cfg.activation_func.__module__ + '.' + cfg.activation_func.__name__,
                        'add_bias_linear': cfg.add_bias_linear,
                        'use_te_activation_func': cfg.use_te_activation_func,
                        'activation_func_fp8_input_store': cfg.activation_func_fp8_input_store,
                        'glu_linear_offset': cfg.glu_linear_offset,
                        'activation_func_clamp_value': cfg.activation_func_clamp_value,
                        'cpu_offloading': cfg.cpu_offloading,
                        'per_token_scale_present': per_token_scale is not None}
            receipt.setdefault('configuration', recorded)
            assert recorded == receipt['configuration']
            observed_configs.add(id(cfg))
        return original_forward(self, hidden_states, per_token_scale, **kwargs)

    def dispatch(x, bias, fp8_input_store=False, cpu_offload_input=False):
        supported = (bias is None and not fp8_input_store and not cpu_offload_input
                     and x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                     and x.ndim in (2, 3) and x.shape[-1] % 2 == 0)
        key = str((tuple(x.shape), tuple(x.stride()), str(x.dtype)))
        receipt['shapes'][key] = receipt['shapes'].get(key, 0) + 1
        if candidate is not None and supported:
            y = candidate.swiglu(x)
            receipt['calls'] += 1
        else:
            if candidate is not None:
                receipt['fallback_calls'] += 1
            receipt['native_calls'] += 1
            y = native(x, bias, fp8_input_store, cpu_offload_input)
        if a.capture_swiglu and not receipt.get('capture_written'):
            a.capture_swiglu.mkdir(exist_ok=False)
            payload = {'input': x.detach().cpu().clone(), 'output': y.detach().cpu().clone()}
            torch.save(payload, a.capture_swiglu / 'forward.pt')
            def save_grad(grad):
                torch.save(grad.detach().cpu().clone(), a.capture_swiglu / 'upstream-grad.pt')
                return grad
            y.register_hook(save_grad)
            receipt['capture_written'] = True
        return y

    if candidate is not None or a.capture_swiglu:
        mlp.bias_swiglu_impl = dispatch
    if a.capture_swiglu or a.swiglu_candidate or '--audit-dir' in rest:
        mlp.MLP.forward = observed_forward
    return result


runtime_compat.prepare_local_runtime = prepare
sys.argv = [str(a.gradnorm_launcher)] + rest
try:
    runpy.run_path(str(a.gradnorm_launcher), run_name='__main__')
finally:
    a.swiglu_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
