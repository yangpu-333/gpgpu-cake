"""Process-local complete-attention dispatch around the unchanged native entry.

Candidate API: attention(query, key, value, softmax_scale).  This controller
contains no attention arithmetic or candidate fallback implementation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--attention-candidate', type=Path)
p.add_argument('--attention-receipt', type=Path, required=True)
p.add_argument('--delegate-launcher', type=Path, required=True)
a, rest = p.parse_known_args()
require(not a.attention_receipt.exists(), 'immutable attention receipt already exists')
observe_native_routes = '--audit-dir' in rest
bf = Path(rest[rest.index('--bf16-root') + 1])
sys.path.insert(0, str(bf.parent / 'megatron_training_entry'))
import runtime_compat
original_prepare = runtime_compat.prepare_local_runtime
receipt = {'schema': 'native-complete-attention-dispatch-v1',
           'enabled': a.attention_candidate is not None, 'calls': 0,
           'candidate_calls': 0, 'native_calls': 0, 'fallback_calls': 0,
           'native_fused_forward_calls': 0, 'native_torch_forward_calls': 0,
           'native_fused_backward_calls': 0,
           'native_routes_observed': observe_native_routes,
           'unsupported_calls': 0, 'shapes': {}, 'configurations': [],
           'wrapper_sha256': sha(Path(__file__)),
           'delegate_launcher_sha256': sha(a.delegate_launcher),
           'candidate_sha256': sha(a.attention_candidate) if a.attention_candidate else None,
           'capture_written': False,
           'scope': 'Whole native DotProductAttention replacement only for the audited causal BF16 scope; '
                    'native/control use original forward, default fused softmax remains enabled. '
                    'Extra native softmax forward/backward route wrappers are installed only for '
                    'untimed audit; performance does not observe those routes per call.'}


def prepare(root):
    result = original_prepare(root)
    import torch
    from megatron.core.fusions import fused_softmax
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    receipt['native_source_sha256'] = {name: sha(Path(root) / name) for name in
        ('megatron/core/transformer/dot_product_attention.py',
         'megatron/core/fusions/fused_softmax.py', 'megatron/core/transformer/utils.py')}
    candidate = None
    if a.attention_candidate:
        spec = importlib.util.spec_from_file_location('stage31_attention_candidate', a.attention_candidate)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        require(callable(getattr(candidate, 'attention', None)), 'candidate attention API missing')

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
        expected = {'num_layers': 4, 'hidden_size': 512, 'num_attention_heads': 8,
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
                     and all(x.ndim == 4 and x.is_cuda and x.dtype == torch.bfloat16
                             and tuple(x.shape) == (128, 2, 8, 64)
                             and tuple(x.stride()) == (3072, 1536, 192, 1)
                             and x.requires_grad for x in (query, key, value))
                     and query.device == key.device == value.device
                     and tuple(x.storage_offset() for x in (query, key, value)) == (0, 64, 128)
                     and attention_mask is not None and attention_mask.is_cuda
                     and attention_mask.dtype == torch.bool
                     and tuple(attention_mask.shape) == (2, 1, 128, 128))
        key_signature = signature(query, key, value, attention_mask)
        receipt['shapes'][key_signature] = receipt['shapes'].get(key_signature, 0) + 1
        if not supported:
            receipt['unsupported_calls'] += 1
        if candidate is not None and supported:
            # The default native causal fused kernel ignores mask values.  The
            # candidate implements the same causal rule, established by probe.
            output = candidate.attention(query, key, value, self.softmax_scale)
            receipt['candidate_calls'] += 1
        else:
            if candidate is not None:
                receipt['fallback_calls'] += 1
            receipt['native_calls'] += 1
            output = native(self, query, key, value, attention_mask, attn_mask_type,
                            attention_bias, packed_seq_params)
        require(output.dtype == torch.bfloat16 and output.is_cuda and output.device == query.device
                and tuple(output.shape) == (128, 2, 512) and output.is_contiguous()
                and output.requires_grad, 'actual complete attention output contract differs')
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
    with a.attention_receipt.open('x') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
