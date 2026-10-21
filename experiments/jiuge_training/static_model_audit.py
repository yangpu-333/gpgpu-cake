"""Inspect public config and safetensors JSON headers without loading weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def audit(root):
    config_path = root / 'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    headers = sorted(root.glob('model-*.safetensors.header.json'))
    tensors, files, errors = {}, {}, []
    for path in [config_path, root / 'configuration_fm9g.py', root / 'modeling_fm9g.py', *headers]:
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in headers:
        for name, metadata in json.loads(path.read_text(encoding='utf-8')).items():
            if name == '__metadata__':
                continue
            if name in tensors:
                errors.append('duplicate tensor name: ' + name)
            tensors[name] = metadata
    if len(headers) != 4:
        errors.append('all four shard JSON headers are required')
    layers = sorted({int(match.group(1)) for name in tensors
                     if (match := re.match(r'model\.layers\.(\d+)\.', name))})
    if layers != list(range(config['num_hidden_layers'])):
        errors.append('header layer set differs from config')
    if any('expert' in name or 'router' in name for name in tensors):
        errors.append('unexpected expert/router weights in the selected Dense reference')
    h, f, v, l, q, kv, d = [config[key] for key in (
        'hidden_size', 'intermediate_size', 'vocab_size', 'num_hidden_layers',
        'num_attention_heads', 'num_key_value_heads', 'head_dim')]
    expected_shapes = {
        'model.embed_tokens.weight': [v, h], 'lm_head.weight': [v, h], 'model.norm.weight': [h]}
    for layer in layers:
        prefix = f'model.layers.{layer}.'
        expected_shapes.update({prefix + name: shape for name, shape in {
            'input_layernorm.weight': [h], 'post_attention_layernorm.weight': [h],
            'self_attn.q_proj.weight': [q * d, h],
            'self_attn.k_proj.weight': [kv * d, h],
            'self_attn.v_proj.weight': [kv * d, h],
            'self_attn.o_proj.weight': [h, q * d],
            'mlp.gate_proj.weight': [f, h], 'mlp.up_proj.weight': [f, h],
            'mlp.down_proj.weight': [h, f],
        }.items()})
    if set(tensors) != set(expected_shapes):
        errors.append('tensor key set differs from the complete Dense/GQA/SwiGLU model')
    for name, shape in expected_shapes.items():
        if tensors.get(name, {}).get('shape') != shape:
            errors.append('shape mismatch: ' + name)
    dtypes = sorted({value['dtype'] for value in tensors.values()})
    if dtypes != ['BF16']:
        errors.append('unexpected tensor dtype')
    parameters = sum(math.prod(value['shape']) for value in tensors.values())
    expected_parameters = 2 * v * h + l * (2 * h * q * d + 2 * h * kv * d + 3 * h * f + 2 * h) + h
    if parameters != expected_parameters:
        errors.append('parameter count formula mismatch')
    return {'schema': 'jiuge-public-metadata-audit-v1', 'public_metadata_consistent': not errors,
            'errors': errors, 'architecture': 'dense', 'model_class': config['architectures'][0],
            'parameter_elements': parameters, 'tensor_count': len(tensors), 'shards': len(headers),
            'model': {'num_layers': l, 'hidden_size': h, 'ffn_hidden_size': f,
                      'num_attention_heads': q, 'num_query_groups': kv, 'head_dim': d,
                      'vocab_size': v, 'rms_norm_eps': config['rms_norm_eps'],
                      'position_embedding_type': config['rope_scaling']['rope_type'],
                      'max_position_embeddings': config['max_position_embeddings'],
                      'scale_emb': config['scale_emb'], 'scale_depth': config['scale_depth'],
                      'dim_model_base': config['dim_model_base'],
                      'residual_branch_multiplier': config['scale_depth'] / math.sqrt(l),
                      'hidden_multiplier_before_lm_head': config['dim_model_base'] / h,
                      'tie_word_embeddings': config['tie_word_embeddings']},
            'metadata_file_sha256': files,
            'moe_grouped_gemm_applicable_to_this_reference': False,
            'selected_ACT_checkpoint_matches_public_reference': None,
            'checkpoint_payload_hash_verified': False, 'model_code_executed': False,
            'gpu_math_or_training_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-root', type=Path,
                        default=Path(__file__).parent / 'results/reference-20261014/archive-metadata')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.metadata_root)
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if result['public_metadata_consistent'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
