"""Map official 7B tensors into native TP shards without changing model math."""
import hashlib
import json
from pathlib import Path


def tensor_sha(tensor):
    data = tensor.detach().contiguous().cpu().view(__import__('torch').uint8).numpy()
    return hashlib.sha256(data.tobytes()).hexdigest()


def shard_mapping(weights, rank, tp, layers=28, hidden=3584, heads=28, ffn=18944, vocab=73448):
    import torch
    if heads % tp or ffn % tp or vocab % tp or hidden % heads:
        raise ValueError('Exact unpadded TP partition required')
    d = hidden // heads
    h0, h1 = rank * heads // tp, (rank + 1) * heads // tp
    f0, f1 = rank * ffn // tp, (rank + 1) * ffn // tp
    v0, v1 = rank * vocab // tp, (rank + 1) * vocab // tp
    yield 'embedding.word_embeddings.weight', weights['model.embed_tokens.weight'][v0:v1], ['model.embed_tokens.weight']
    yield 'output_layer.weight', weights['lm_head.weight'][v0:v1], ['lm_head.weight']
    yield 'decoder.final_layernorm.weight', weights['model.norm.weight'], ['model.norm.weight']
    for index in range(layers):
        source, target = f'model.layers.{index}.', f'decoder.layers.{index}.'
        for origin, name in (('input_layernorm.weight', 'input_layernorm.weight'),
                             ('post_attention_layernorm.weight', 'pre_mlp_layernorm.weight')):
            yield target + name, weights[source + origin], [source + origin]
        for suffix in ('weight', 'bias'):
            keys = [source + 'self_attn.' + part + '_proj.' + suffix for part in ('q', 'k', 'v')]
            shape = (heads, d, hidden) if suffix == 'weight' else (heads, d)
            values = [weights[key].reshape(shape)[h0:h1] for key in keys]
            # Native MHA QKV rows are ordered per head: Q_h, K_h, V_h.
            combined = torch.cat(values, dim=1).flatten(0, 1)
            yield target + 'self_attention.linear_qkv.' + suffix, combined, keys
        yield target + 'self_attention.linear_proj.weight', weights[source + 'self_attn.o_proj.weight'][:, h0*d:h1*d], [source + 'self_attn.o_proj.weight']
        keys = [source + 'mlp.' + part + '_proj.weight' for part in ('gate', 'up')]
        # TP stride=2 partitions gate and up separately before concatenation.
        yield target + 'mlp.linear_fc1.weight', torch.cat([weights[key][f0:f1] for key in keys], dim=0), keys
        yield target + 'mlp.linear_fc2.weight', weights[source + 'mlp.down_proj.weight'][:, f0:f1], [source + 'mlp.down_proj.weight']


def load_into_native(model, checkpoint, output):
    import torch
    from megatron.core import parallel_state
    cfg = model.config
    required = {'num_layers': 28, 'hidden_size': 3584, 'ffn_hidden_size': 18944,
                'num_attention_heads': 28, 'num_query_groups': 28,
                'normalization': 'RMSNorm', 'layernorm_epsilon': 1e-6,
                'add_bias_linear': False, 'add_qkv_bias': True}
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError('7B model semantic mismatch: ' + key)
    if model.vocab_size != 73448 or model.share_embeddings_and_output_weights:
        raise ValueError('Exact untied 7B vocabulary required')
    if parallel_state.get_pipeline_model_parallel_world_size() != 1:
        raise ValueError('This bridge only admits PP1')
    tp, rank = parallel_state.get_tensor_model_parallel_world_size(), parallel_state.get_tensor_model_parallel_rank()
    weights = torch.load(checkpoint / 'pytorch_model.bin', mmap=True, map_location='cpu', weights_only=True)
    if len(weights) != 339 or sum(value.numel() for value in weights.values()) != 7668831744:
        raise ValueError('Complete official 7B weights required')
    parameters = dict(model.named_parameters())
    consumed, loaded, rows = set(), set(), []
    with torch.no_grad():
        for name, value, keys in shard_mapping(weights, rank, tp):
            if name not in parameters or list(parameters[name].shape) != list(value.shape):
                raise ValueError('Native shard shape/key mismatch: ' + name)
            parameter = parameters[name]
            expected = value.to(dtype=parameter.dtype).contiguous()
            parameter.copy_(expected)
            expected_sha, actual_sha = tensor_sha(expected), tensor_sha(parameter)
            if actual_sha != expected_sha:
                raise ValueError('Loaded shard bytes differ: ' + name)
            rows.append({'native_name': name, 'source_names': keys, 'shape': list(parameter.shape),
                         'dtype': str(parameter.dtype), 'sha256': actual_sha})
            consumed.update(keys)
            loaded.add(name)
    if consumed != set(weights) or loaded != set(parameters):
        raise ValueError('Every source tensor and native parameter must be covered')
    record = {'schema': 'jiuge7b-native-shard-loading-v1', 'tp_rank': rank, 'tp_size': tp,
              'source_tensor_count': len(weights), 'source_parameter_elements': 7668831744,
              'native_parameter_count': len(parameters), 'native_shard_elements': sum(p.numel() for p in parameters.values()),
              'all_source_tensors_and_native_parameters_covered': True,
              'all_loaded_shard_bytes_verified': True, 'parameters': rows,
              'model_math_modified': False, 'extra_8b_scaling_applied': False,
              'bridge_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open('x') as stream:
        json.dump(record, stream, indent=2)
    del weights
    return record


def compare_forward_reference(model, reference, output):
    import torch
    from megatron.core import parallel_state, tensor_parallel
    if not (reference / 'receipt.json').is_file():
        raise ValueError('Complete official forward reference required')
    original_training = model.training
    config = model.config
    original_precision = {key:getattr(config,key) for key in ('params_dtype','bf16','fp16')}
    softmax_precision = []
    for module in model.modules():
        for key in ('input_in_fp16','input_in_bf16','input_in_float16'):
            if hasattr(module,key):
                softmax_precision.append((module,key,getattr(module,key)))
                setattr(module,key,False)
    config.params_dtype, config.bf16, config.fp16 = torch.float32, False, False
    model.cuda().float().eval()
    rows = []
    try:
        with torch.no_grad():
            for index in range(3):
                case = torch.load(reference / f'case-{index}.pt', map_location='cpu', weights_only=True)
                tokens = case['tokens'].cuda()
                positions = torch.arange(tokens.shape[1], device=tokens.device)[None].expand_as(tokens)
                value = model(tokens, positions, None)
                value = tensor_parallel.gather_from_tensor_model_parallel_region(value)
                actual, expected = value.detach().cpu().double(), case['logits'].double()
                difference = (actual - expected).abs()
                passed = torch.allclose(actual, expected, atol=3e-4, rtol=1e-3)
                loss = torch.nn.functional.cross_entropy(value[:,:-1].float().flatten(0,1),tokens[:,1:].flatten())
                rows.append({'case':index,'sequence':tokens.shape[1], 'passed':passed,
                             'max_abs':difference.max().item(), 'mean_abs':difference.mean().item(),
                             'native_loss':loss.item(),'official_loss':case['loss'].item()})
                if parallel_state.get_tensor_model_parallel_rank() == 0:
                    torch.save({'logits':actual.float(),'tokens':case['tokens'],'loss':loss.cpu()},
                               output.parent/f'native-forward-{index}.pt')
                print(json.dumps({'forward_comparison':rows[-1]}),flush=True)
    finally:
        model.bfloat16().train(original_training)
        for key,value in original_precision.items():
            setattr(config,key,value)
        for module,key,value in softmax_precision:
            setattr(module,key,value)
    record = {'schema':'jiuge7b-native-forward-equivalence-v1','cases':rows,
              'comparison':'FP32 forward from identical original BF16 weights',
              'atol':3e-4,'rtol':1e-3,'comparison_dtype':'CPU float64',
              'bf16_training_update_equivalence_claimed':False,
              'passed':all(row['passed'] for row in rows)}
    with output.open('x') as stream:
        json.dump(record,stream,indent=2)
    if not record['passed']:
        raise ValueError('7B forward equivalence did not pass the frozen tolerance')
    return record
