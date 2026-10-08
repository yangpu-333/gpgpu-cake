"""Stage32 process-only attention dispatch; no candidate or native mathematics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


CANDIDATE_SHA = '4881f91c230324f258cf6faf7b1d63a493b043b960df4efdca0dad708195db48'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--attention-candidate', type=Path)
p.add_argument('--attention-receipt', type=Path, required=True)
p.add_argument('--swiglu-receipt', type=Path, required=True)
p.add_argument('--delegate-launcher', type=Path, required=True)
a, rest = p.parse_known_args()
require(not a.attention_receipt.exists() and not a.swiglu_receipt.exists(), 'immutable receipt already exists')
require('--swiglu-candidate' not in rest and '--capture-swiglu' not in rest,
        'SwiGLU candidate/capture is forbidden in all stage32 arms')
contract_path = Path(rest[rest.index('--contract') + 1])
contract = json.loads(contract_path.read_text(encoding='utf-8'))
model = contract['model']
layers, sequence = model['num_layers'], model['sequence_length']
require((layers, sequence, model['vocab_size'], model['parameters']) in
        ((4, 256, 4096, 21107200), (8, 512, 8192, 42213888)), 'unsupported fixed held-out model')
require(model['micro_batch_size'] == model['global_batch_size'] == 2
        and model['hidden_size'] == 512 and model['ffn_hidden_size'] == 2048, 'fixed shape differs')
observe_native_routes = '--audit-dir' in rest
bf = Path(rest[rest.index('--bf16-root') + 1])
sys.path.insert(0, str(bf.parent / 'megatron_training_entry'))
import runtime_compat
original_prepare = runtime_compat.prepare_local_runtime
receipt = {'schema': 'native-complete-attention-stage32-dispatch-v1',
           'enabled': a.attention_candidate is not None, 'calls': 0,
           'candidate_calls': 0, 'native_calls': 0, 'fallback_calls': 0,
           'native_fused_forward_calls': 0, 'native_torch_forward_calls': 0,
           'native_fused_backward_calls': 0, 'native_routes_observed': observe_native_routes,
           'unsupported_calls': 0, 'shapes': {}, 'configurations': [], 'candidate_jit': {},
           'candidate_backend_cache': [], 'candidate_scale_mode': None,
           'wrapper_sha256': sha(Path(__file__)), 'delegate_launcher_sha256': sha(a.delegate_launcher),
           'candidate_sha256': sha(a.attention_candidate) if a.attention_candidate else None,
           'capture_written': False,
           'scope': 'Frozen Attention003 only for declared causal BF16 larger models; native/control '
                    'use the original default fused forward. Extra original softmax route wrappers '
                    'are installed only in untimed audit, never during performance. Candidate cache '
                    'metadata and JIT graphs identify the frozen implementation; no GPU timeline claim.'}
activation = {'schema': 'native-default-swiglu-stage32-observation-v1',
              'enabled': False, 'candidate_sha256': None, 'candidate_calls': 0,
              'calls': 0, 'native_calls': 0, 'fallback_calls': 0, 'capture_written': False,
              'native_calls_observed': observe_native_routes,
              'configuration_observed': observe_native_routes,
              'shapes': {}, 'configurations': [], 'extension': None,
              'wrapper_sha256': sha(Path(__file__)),
              'scope': 'Original default compiled bias_swiglu_impl in every arm. Actual call/config '
                       'observation only in untimed audit; no SwiGLU wrapper during performance.'}
candidate_module = None


def prepare(root):
    global candidate_module
    result = original_prepare(root)
    import torch
    from megatron.core.fusions import fused_softmax
    import megatron.core.fusions.fused_bias_swiglu as fused_swiglu
    import megatron.core.transformer.mlp as mlp
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    receipt['native_source_sha256'] = {name: sha(Path(root) / name) for name in
        ('megatron/core/transformer/dot_product_attention.py',
         'megatron/core/fusions/fused_softmax.py', 'megatron/core/transformer/utils.py')}
    activation['native_source_sha256'] = {name: sha(Path(root) / name) for name in
        ('megatron/core/transformer/mlp.py',
         'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    activation['native_jit_callable'] = str(type(fused_swiglu.swiglu))
    activation['native_compiled_callable'] = hasattr(fused_swiglu.swiglu, '_torchdynamo_orig_callable')
    require(activation['native_compiled_callable'], 'original SwiGLU compile backend unavailable')
    if a.attention_candidate:
        require(sha(a.attention_candidate) == CANDIDATE_SHA, 'Attention003 bytes may not change')
        spec = importlib.util.spec_from_file_location('stage32_frozen_attention003', a.attention_candidate)
        candidate_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate_module)
        require(callable(getattr(candidate_module, 'attention', None)), 'Attention API missing')
        for name in ('_fwd_scores', '_fwd_context', '_bwd_context', '_bwd_scores'):
            function = getattr(candidate_module, name, None)
            scripted = isinstance(function, torch.jit.ScriptFunction)
            graph = str(function.graph) if scripted else None
            receipt['candidate_jit'][name] = {
                'script_function': scripted, 'type': str(type(function)), 'graph': graph,
                'graph_sha256': hashlib.sha256(graph.encode('utf-8')).hexdigest() if scripted else None}
        require(all(row['script_function'] for row in receipt['candidate_jit'].values()),
                'Attention003 silently fell back to eager stage functions')

    native = DotProductAttention.forward
    configurations = {}
    original_fused = fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax
    original_torch = fused_softmax.FusedScaleMaskSoftmax.forward_torch_softmax
    original_backward = fused_softmax.ScaledUpperTriangMaskedSoftmax.backward

    def observe_fused(self, inputs, mask):
        receipt['native_fused_forward_calls'] += 1
        return original_fused(self, inputs, mask)

    def observe_torch(self, inputs, mask, softmax_offset=None):
        receipt['native_torch_forward_calls'] += 1
        return original_torch(self, inputs, mask, softmax_offset)

    def observe_backward(ctx, output_grads):
        receipt['native_fused_backward_calls'] += 1
        return original_backward(ctx, output_grads)

    if observe_native_routes:
        fused_softmax.FusedScaleMaskSoftmax.forward_fused_softmax = observe_fused
        fused_softmax.FusedScaleMaskSoftmax.forward_torch_softmax = observe_torch
        fused_softmax.ScaledUpperTriangMaskedSoftmax.backward = staticmethod(observe_backward)

        native_swiglu = mlp.bias_swiglu_impl
        original_mlp = mlp.MLP.forward
        observed_configs = set()

        def observed_mlp(self, hidden_states, per_token_scale=None, **kwargs):
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
                if recorded not in activation['configurations']:
                    activation['configurations'].append(recorded)
                observed_configs.add(id(cfg))
            return original_mlp(self, hidden_states, per_token_scale, **kwargs)

        def observed_swiglu(x, bias, fp8_input_store=False, cpu_offload_input=False):
            require(bias is None and not fp8_input_store and not cpu_offload_input
                    and x.is_cuda and x.dtype == torch.bfloat16 and x.is_contiguous()
                    and tuple(x.shape) == (sequence, 2, 4096), 'actual native SwiGLU scope differs')
            activation['calls'] += 1
            activation['native_calls'] += 1
            key = str((tuple(x.shape), tuple(x.stride()), str(x.dtype)))
            activation['shapes'][key] = activation['shapes'].get(key, 0) + 1
            return native_swiglu(x, bias, fp8_input_store, cpu_offload_input)

        mlp.MLP.forward = observed_mlp
        mlp.bias_swiglu_impl = observed_swiglu

    def configuration(self, mask, query, key):
        cfg, softmax = self.config, self.scale_mask_softmax
        recorded = {name: getattr(cfg, name) for name in
            ('num_layers', 'hidden_size', 'num_attention_heads', 'num_query_groups',
             'kv_channels', 'attention_dropout', 'masked_softmax_fusion',
             'attention_softmax_in_fp32', 'apply_query_key_layer_scaling',
             'sequence_parallel', 'context_parallel_size', 'tensor_model_parallel_size',
             'softmax_type', 'window_size', 'bf16', 'fp16')}
        recorded.update(softmax_scale=self.softmax_scale,
                        softmax_offset_present=self.softmax_offset is not None,
                        attn_mask_type=str(self.attn_mask_type),
                        dropout_probability=self.attention_dropout.p,
                        tp_group_size=self.tp_group.size(),
                        heads_per_partition=self.num_attention_heads_per_partition,
                        groups_per_partition=self.num_query_groups_per_partition,
                        head_dimension=self.hidden_size_per_attention_head,
                        hidden_per_partition=self.hidden_size_per_partition,
                        scaled_masked_softmax_fusion=softmax.scaled_masked_softmax_fusion,
                        softmax_in_fp32=softmax.softmax_in_fp32,
                        additional_softmax_scale=softmax.scale,
                        softmax_window_size=softmax.window_size,
                        softmax_attn_mask_type=str(softmax.attn_mask_type),
                        softmax_input_bf16=softmax.input_in_bf16,
                        softmax_input_fp16=softmax.input_in_fp16)
        recorded['native_kernel_available'] = bool(softmax.is_kernel_available(
            mask, query.shape[1], query.shape[2], query.shape[0], key.shape[0]))
        return recorded

    def supported_configuration(row):
        expected = {'num_layers': layers, 'hidden_size': 512, 'num_attention_heads': 8,
                    'num_query_groups': 8, 'kv_channels': 64, 'attention_dropout': 0.,
                    'masked_softmax_fusion': True, 'attention_softmax_in_fp32': False,
                    'apply_query_key_layer_scaling': False, 'sequence_parallel': False,
                    'context_parallel_size': 1, 'tensor_model_parallel_size': 1,
                    'softmax_type': 'vanilla', 'window_size': None, 'bf16': True, 'fp16': False,
                    'softmax_scale': .125, 'softmax_offset_present': False,
                    'attn_mask_type': str(AttnMaskType.causal), 'dropout_probability': 0.,
                    'tp_group_size': 1, 'heads_per_partition': 8, 'groups_per_partition': 8,
                    'head_dimension': 64, 'hidden_per_partition': 512,
                    'scaled_masked_softmax_fusion': True, 'softmax_in_fp32': False,
                    'additional_softmax_scale': None, 'softmax_window_size': None,
                    'softmax_attn_mask_type': str(AttnMaskType.causal),
                    'softmax_input_bf16': True, 'softmax_input_fp16': False,
                    'native_kernel_available': True}
        return row == expected

    def signature(query, key, value, mask):
        values = [tuple((tuple(x.shape), tuple(x.stride()), str(x.dtype),
                         x.storage_offset(), x.requires_grad, x.is_cuda)) for x in (query, key, value)]
        values.append(None if mask is None else
                      (tuple(mask.shape), tuple(mask.stride()), str(mask.dtype),
                       mask.storage_offset(), mask.requires_grad, mask.is_cuda))
        return str(tuple(values))

    def forward(self, query, key, value, attention_mask, attn_mask_type=None,
                attention_bias=None, packed_seq_params=None):
        receipt['calls'] += 1
        if id(self) not in configurations:
            row = configuration(self, attention_mask, query, key)
            configurations[id(self)] = supported_configuration(row)
            receipt['configurations'].append(row)
        supported = (configurations[id(self)] and attention_bias is None and packed_seq_params is None
                     and attn_mask_type in (None, AttnMaskType.causal)
                     and all(x.ndim == 4 and x.is_cuda and x.dtype == torch.bfloat16
                             and tuple(x.shape) == (sequence, 2, 8, 64)
                             and tuple(x.stride()) == (3072, 1536, 192, 1)
                             and x.requires_grad for x in (query, key, value))
                     and query.device == key.device == value.device
                     and tuple(x.storage_offset() for x in (query, key, value)) == (0, 64, 128)
                     and attention_mask is not None and attention_mask.is_cuda
                     and attention_mask.dtype == torch.bool and not attention_mask.requires_grad
                     and attention_mask.storage_offset() == 0
                     and tuple(attention_mask.shape) == (2, 1, sequence, sequence)
                     and tuple(attention_mask.stride()) == (sequence ** 2, sequence ** 2, sequence, 1))
        key_signature = signature(query, key, value, attention_mask)
        receipt['shapes'][key_signature] = receipt['shapes'].get(key_signature, 0) + 1
        if not supported:
            receipt['unsupported_calls'] += 1
        if candidate_module is not None and supported:
            output = candidate_module.attention(query, key, value, self.softmax_scale)
            receipt['candidate_calls'] += 1
        else:
            if candidate_module is not None:
                receipt['fallback_calls'] += 1
            receipt['native_calls'] += 1
            output = native(self, query, key, value, attention_mask, attn_mask_type,
                            attention_bias, packed_seq_params)
        require(output.dtype == torch.bfloat16 and output.is_cuda and output.device == query.device
                and tuple(output.shape) == (sequence, 2, 512) and output.is_contiguous()
                and output.requires_grad, 'actual complete Attention output contract differs')
        return output

    DotProductAttention.forward = forward
    return result


runtime_compat.prepare_local_runtime = prepare
sys.argv = [str(a.delegate_launcher)] + rest
try:
    runpy.run_path(str(a.delegate_launcher), run_name='__main__')
finally:
    receipt['native_extensions'] = {}
    for name in ('scaled_masked_softmax_cuda', 'scaled_upper_triang_masked_softmax_cuda'):
        module = sys.modules.get(name)
        if module is not None:
            path = Path(module.__file__).resolve()
            receipt['native_extensions'][name] = {'path': str(path), 'sha256': sha(path)}
    if candidate_module is not None:
        receipt['candidate_backend_cache'] = [
            {'shape': list(shape), 'batch_per_block': value}
            for shape, value in sorted(candidate_module._BPB_CACHE.items())]
        receipt['candidate_scale_mode'] = candidate_module._SCALE_MODE
    for path, value in ((a.attention_receipt, receipt), (a.swiglu_receipt, activation)):
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
