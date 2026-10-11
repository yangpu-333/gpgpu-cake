"""Stage34 weighted expert activation hook; native source/math remain the oracle."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--weighted-receipt', type=Path, required=True)
p.add_argument('--weighted-candidate', type=Path)
p.add_argument('--capture-limit', type=int, default=0)
p.add_argument('--route-capture', action='store_true')
p.add_argument('--bf16-root', type=Path, required=True)
a, rest = p.parse_known_args()
assert not a.weighted_receipt.exists()
sys.path.insert(0, str(a.bf16_root.parent / 'megatron_training_entry'))
import runtime_compat
original_prepare = runtime_compat.prepare_local_runtime
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt = {'schema': 'moe-weighted-stage34-dispatch-v1', 'enabled': a.weighted_candidate is not None,
    'candidate_sha256': sha(a.weighted_candidate) if a.weighted_candidate else None,
    'wrapper_sha256': sha(__file__), 'calls': 0, 'candidate_calls': 0, 'native_calls': 0,
    'fallback_calls': 0, 'sort_compat_calls': 0, 'shapes': {}, 'expert_calls': [], 'router_calls': 0,
    'captures': [], 'configurations': [], 'raw_files': {},
    'scope': 'Native SequentialMLP weighted_bias_swiglu call only. Common explicit bool-to0/1-uint8 stable '
             'sort compatibility; no candidate GEMM, RMSNorm, attention or router changes. '
             'Raw tensors/config/token/probability-gradient observation only in untimed audit/capture.'}


def prepare(root):
    result = original_prepare(root)
    import torch
    import megatron.core.transformer.mlp as mlp
    import megatron.core.fusions.fused_bias_swiglu as fusion
    from megatron.core.transformer.moe.router import TopKRouter
    from megatron.core.transformer.moe.experts import SequentialMLP
    native_sort = torch.Tensor.argsort
    def compatible_sort(tensor, *args, **kwargs):
        if tensor.is_cuda and tensor.dtype == torch.bool and kwargs.get('stable') is True:
            receipt['sort_compat_calls'] += 1
            return native_sort(tensor.to(torch.uint8), *args, **kwargs)
        return native_sort(tensor, *args, **kwargs)
    torch.Tensor.argsort = compatible_sort
    receipt['native_source_sha256'] = {name: sha(Path(root) / name) for name in (
        'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/transformer/mlp.py',
        'megatron/core/transformer/moe/experts.py', 'megatron/core/transformer/moe/router.py',
        'megatron/core/transformer/moe/token_dispatcher.py', 'megatron/core/transformer/moe/moe_utils.py')}
    receipt['native_compiled_callables'] = {name: hasattr(getattr(fusion, name), '_torchdynamo_orig_callable')
                                           for name in ('weighted_swiglu', 'weighted_swiglu_back')}
    native = mlp.weighted_bias_swiglu_impl
    module = None
    if a.weighted_candidate:
        spec = importlib.util.spec_from_file_location('stage34_weighted_candidate', a.weighted_candidate)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    base = a.weighted_receipt.parent
    captured = {}
    def save_capture(index):
        value = captured[index]
        if all(key in value for key in ('input_grad', 'weights_grad', 'upstream')):
            path = base / 'activation' / ('capture-%03d.pt' % index)
            path.parent.mkdir(exist_ok=True)
            assert not path.exists()
            torch.save(value, path)
    def call(x, bias, weights, fp8_input_store=False):
        receipt['calls'] += 1
        key = str((tuple(x.shape), str(x.dtype), tuple(weights.shape), str(weights.dtype)))
        receipt['shapes'][key] = receipt['shapes'].get(key, 0) + 1
        index = receipt['calls']
        capture = index <= a.capture_limit
        if capture:
            captured[index] = {'input': x.detach().cpu().clone(), 'weights': weights.detach().cpu().clone()}
            def observe(label):
                def hook(grad):
                    captured[index][label] = grad.detach().cpu().clone()
                    save_capture(index)
                    return grad
                return hook
            x.register_hook(observe('input_grad'))
            weights.register_hook(observe('weights_grad'))
        def fallback(xx, ww):
            receipt['fallback_calls'] += 1
            return native(xx, bias, ww, fp8_input_store)
        if module is not None:
            assert bias is None and fp8_input_store is False
            receipt['candidate_calls'] += 1
            y = module.weighted_swiglu(x, weights, native=fallback)
        else:
            receipt['native_calls'] += 1
            y = native(x, bias, weights, fp8_input_store)
        if capture:
            captured[index]['output'] = y.detach().cpu().clone()
            y.register_hook(observe('upstream'))
        return y
    mlp.weighted_bias_swiglu_impl = call
    # Extra observations are deliberately absent during performance runs.
    if a.route_capture:
        original_router, original_experts = TopKRouter.forward, SequentialMLP.forward
        route_root = base / 'routing'; route_root.mkdir()
        def router(self, *args, **kwargs):
            probabilities, routing_map = original_router(self, *args, **kwargs)
            receipt['router_calls'] += 1
            number = receipt['router_calls']
            torch.save({'probabilities': probabilities.detach().cpu().clone(),
                        'routing_map': routing_map.detach().cpu().clone()},
                       route_root / ('router-%03d.pt' % number))
            def gradient(grad):
                torch.save(grad.detach().cpu().clone(), route_root / ('probability-grad-%03d.pt' % number))
                return grad
            probabilities.register_hook(gradient)
            return probabilities, routing_map
        def experts(self, hidden, counts, probabilities):
            cfg = self.config
            configuration = {name: getattr(cfg, name, None) for name in
                ('num_moe_experts', 'moe_router_topk', 'moe_grouped_gemm', 'moe_token_dispatcher_type',
                 'moe_apply_probs_on_input', 'moe_aux_loss_coeff', 'gated_linear_unit',
                 'bias_activation_fusion', 'masked_softmax_fusion', 'use_te_activation_func')}
            if configuration not in receipt['configurations']:
                receipt['configurations'].append(configuration)
            receipt['expert_calls'].append({'class': type(self).__name__,
                'tokens_per_expert': counts.detach().cpu().tolist(), 'hidden_shape': list(hidden.shape),
                'probability_dtype': str(probabilities.dtype)})
            return original_experts(self, hidden, counts, probabilities)
        TopKRouter.forward, SequentialMLP.forward = router, experts
    return result


runtime_compat.prepare_local_runtime = prepare
delegate = Path('/tmp/kda-gradnorm-stage26/launch_gradnorm-04.py')
sys.argv = [str(delegate), '--bf16-root', str(a.bf16_root)] + rest
try:
    runpy.run_path(str(delegate), run_name='__main__')
finally:
    for folder in ('activation', 'routing'):
        for path in (a.weighted_receipt.parent / folder).glob('*.pt'):
            receipt['raw_files'][path.relative_to(a.weighted_receipt.parent).as_posix()] = {
                'bytes': path.stat().st_size, 'sha256': sha(path)}
    a.weighted_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
