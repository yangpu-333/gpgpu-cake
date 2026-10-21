"""Check preparation completeness; never load weights or start GPU work."""
import argparse
import json
from pathlib import Path
import re

COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
FROZEN_BENCHMARK = {
    'correctness_steps': 3, 'comparison_dtype': 'CPU float64',
    'atol': .0003, 'rtol': .001,
    'formal_timer': 'native GPU-synchronized time.time',
    'steps_per_process': 120, 'excluded_initial_steps': 20,
    'retained_interval_steps': 10, 'independent_seeds': 3,
    'arms': ['native', 'control', 'optimized'],
    'rotated_arm_orders': [['native', 'control', 'optimized'],
                          ['optimized', 'control', 'native'],
                          ['control', 'native', 'optimized']],
    'promotion_gain_vs_control_strictly_greater_than': .01,
    'all_seeds_faster_vs_control_required': True,
    'native_or_control_cv_repeat_trigger': .1,
    'full_raw_backup_before_formal_timing_required': True,
    'audit_profile_checkpoint_backup_excluded_from_timing': True,
    'resource_gates_before_and_after_required': True,
}


def resident_adamw_estimate(parameters):
    """BF16 weights + FP32 gradients/masters/moments, before activations."""
    sizes = {'bf16_weights': 2, 'fp32_gradients': 4,
             'fp32_master_weights': 4, 'adamw_moments': 8}
    return {'assumption': 'all states resident, unsharded, FP32 main gradients',
            'parameter_count': parameters,
            'bytes_by_component': {key: parameters * value for key, value in sizes.items()},
            'total_bytes': parameters * sum(sizes.values()),
            'total_gib': parameters * sum(sizes.values()) / 2 ** 30,
            'excludes_activations_temporary_buffers_and_communication': True,
            'measured_peak_memory': False}


def check_contract(data):
    issues = []
    if data.get('schema') != 'jiuge-kda-training-v1':
        issues.append('unsupported contract schema')
    if data.get('framework', {}).get('commit') != COMMIT:
        issues.append('native Megatron commit changed')
    if data.get('benchmark') != FROZEN_BENCHMARK:
        issues.append('benchmark differs from the previous frozen standard')

    def require(section, keys, prefix):
        for key in keys:
            value = section.get(key)
            if value is None or value == '' or value == []:
                issues.append(prefix + '.' + key + ' is not fixed')
            elif key.endswith('sha256') and not re.fullmatch('[0-9a-f]{64}', str(value)):
                issues.append(prefix + '.' + key + ' is not a SHA256 digest')

    framework = data.get('framework', {})
    require(framework, ['checkout_path', 'source_manifest_sha256', 'native_backend',
                        'conversion_code_sha256'], 'framework')
    model = data.get('model', {})
    require(model, ['checkpoint_variant', 'checkpoint_path', 'checkpoint_manifest_sha256',
                    'config_path', 'config_sha256', 'tokenizer_path',
                    'tokenizer_manifest_sha256', 'architecture', 'num_parameters',
                    'num_layers', 'hidden_size', 'ffn_hidden_size', 'num_attention_heads',
                    'num_query_groups', 'vocab_size', 'position_embedding_type'], 'model')
    if model.get('architecture') not in ('dense', 'moe'):
        issues.append('actual model architecture is not identified')
    count = model.get('num_parameters')
    if not isinstance(count, int) or isinstance(count, bool) or count < 1_000_000_000:
        issues.append('real billion-scale Jiuge weights are not identified')
    if model.get('checkpoint_conversion_and_loading_verified') is not True:
        issues.append('checkpoint conversion/loading equivalence is unverified')
    runtime = data.get('runtime', {})
    require(runtime, ['corex_version', 'torch_version', 'triton_version',
                      'container_identifier', 'persistent_evidence_root'], 'runtime')
    topology = data.get('topology', {})
    values = [topology.get('pilot_' + key) for key in ('tp', 'pp', 'dp')]
    if not all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in values):
        issues.append('pilot TP/PP/DP is not fixed')
    else:
        world = values[0] * values[1] * values[2]
        available = topology.get('available_gpus')
        if not isinstance(available, int) or world > available:
            issues.append('pilot topology exceeds verified available GPUs')
        hidden, heads, groups = (model.get(key) for key in
                                 ('hidden_size', 'num_attention_heads', 'num_query_groups'))
        if all(isinstance(v, int) and v > 0 for v in (hidden, heads, groups)):
            # The pinned Megatron supports replicated KV groups when TP > GQA groups.
            # Its transformer_config requires either groups % TP == 0 or TP % groups == 0.
            incompatible_groups = groups % values[0] and values[0] % groups
            if hidden % heads or heads % groups or heads % values[0] or incompatible_groups:
                issues.append('head/GQA partitioning needs explicit Megatron compatibility proof')
    require(topology, ['memory_strategy'], 'topology')
    if topology.get('fit_verified') is not True:
        issues.append('native complete-step memory fit is unverified')
    required_task = ['dataset_path', 'dataset_manifest_sha256', 'preprocessing_code_sha256',
                     'sequence_length', 'micro_batch_size', 'global_batch_size', 'optimizer',
                     'optimizer_config_sha256', 'precision', 'activation_recompute',
                     'offload_strategy', 'training_argv']
    for name in ('continued_pretraining', 'sft'):
        task = data.get('tasks', {}).get(name, {})
        require(task, required_task, 'tasks.' + name)
        if task.get('mock_data') is not False or '--mock-data' in (task.get('training_argv') or []):
            issues.append(name + ' uses mock data')
        if name == 'sft':
            require(task, ['loss_mask_code_sha256'], 'tasks.sft')
            if task.get('method') != 'full_parameter_sft':
                issues.append('the selected SFT scenario is not full-parameter SFT')
        batch = [task.get('micro_batch_size'), task.get('global_batch_size')]
        if all(isinstance(v, int) and v > 0 for v in batch) and isinstance(values[2], int) and values[2] > 0:
            if batch[1] % (batch[0] * values[2]):
                issues.append(name + ' global batch is inconsistent with micro-batch and DP')
    return {'configuration_ready': not issues, 'issues': issues,
            'starts_training': False, 'checks_weight_bytes_or_gpu_math': False,
            'nominal_8b_adamw_memory_estimate': resident_adamw_estimate(8_000_000_000)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=Path(__file__).with_name('contract.template.json'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = check_contract(json.loads(args.contract.read_text(encoding='utf-8')))
    text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text, end='')
    return 0 if result['configuration_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
