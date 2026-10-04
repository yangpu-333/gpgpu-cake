"""Untimed native attention capture; arguments and native tensors pass unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
p.add_argument('--attention-capture', type=Path, required=True)
p.add_argument('--attention-receipt', type=Path, required=True)
p.add_argument('--delegate-launcher', type=Path, required=True)
a, rest = p.parse_known_args()
assert not a.attention_receipt.exists() and not a.attention_capture.exists()
bf = Path(rest[rest.index('--bf16-root') + 1])
sys.path.insert(0, str(bf.parent / 'megatron_training_entry'))
import runtime_compat
original_prepare = runtime_compat.prepare_local_runtime
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt = {'scope': 'untimed native attention input/output/gradient capture; no candidate',
           'calls': 0, 'capture_written': False, 'wrapper_sha256': sha(Path(__file__)),
           'observations': []}


def prepare(root):
    result = original_prepare(root)
    import torch
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    native = DotProductAttention.forward
    receipt['native_source_sha256'] = {name: sha(Path(root) / name) for name in
        ('megatron/core/transformer/dot_product_attention.py',
         'megatron/core/fusions/fused_softmax.py',
         'megatron/core/transformer/utils.py')}
    def meta(x):
        return None if x is None else {'shape': list(x.shape), 'stride': list(x.stride()),
             'dtype': str(x.dtype), 'storage_offset': x.storage_offset(),
             'requires_grad': x.requires_grad}
    def observed(self, query, key, value, attention_mask, attn_mask_type=None,
                 attention_bias=None, packed_seq_params=None):
        receipt['calls'] += 1
        first = not receipt['capture_written']
        if first:
            a.attention_capture.mkdir(parents=True, exist_ok=False)
            torch.save({name: x.detach().cpu().clone() if x is not None else None
                        for name, x in [('query', query), ('key', key), ('value', value),
                                        ('attention_mask', attention_mask)]},
                       a.attention_capture / 'inputs.pt')
            for name, x in [('query', query), ('key', key), ('value', value)]:
                def save(grad, label=name):
                    torch.save(grad.detach().cpu().clone(), a.attention_capture / (label+'-grad.pt'))
                    return grad
                x.register_hook(save)
        y = native(self, query, key, value, attention_mask, attn_mask_type,
                   attention_bias, packed_seq_params)
        if first:
            torch.save(y.detach().cpu().clone(), a.attention_capture / 'output.pt')
            def save_upstream(grad):
                torch.save(grad.detach().cpu().clone(), a.attention_capture / 'upstream-grad.pt')
                return grad
            y.register_hook(save_upstream)
            cfg = self.config
            receipt['observations'].append({'query': meta(query), 'key': meta(key),
                'value': meta(value), 'attention_mask': meta(attention_mask), 'output': meta(y),
                'softmax_scale': self.softmax_scale, 'softmax_offset': self.softmax_offset,
                'attn_mask_type': str(self.attn_mask_type),
                'configuration': {name: getattr(cfg, name) for name in
                    ('num_layers', 'hidden_size', 'num_attention_heads', 'num_query_groups',
                     'kv_channels', 'attention_dropout', 'masked_softmax_fusion',
                     'attention_softmax_in_fp32', 'apply_query_key_layer_scaling',
                     'sequence_parallel', 'context_parallel_size', 'softmax_type')},
                'scaled_masked_softmax_fusion': self.scale_mask_softmax.scaled_masked_softmax_fusion,
                'softmax_in_fp32': self.scale_mask_softmax.softmax_in_fp32})
            receipt['capture_written'] = True
        return y
    DotProductAttention.forward = observed
    return result


runtime_compat.prepare_local_runtime = prepare
sys.argv = [str(a.delegate_launcher)] + rest
try:
    runpy.run_path(str(a.delegate_launcher), run_name='__main__')
finally:
    if a.attention_capture.exists():
        receipt['raw_files'] = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}
                                for path in a.attention_capture.glob('*.pt')}
    a.attention_receipt.write_text(json.dumps(receipt, indent=2) + '\n')
