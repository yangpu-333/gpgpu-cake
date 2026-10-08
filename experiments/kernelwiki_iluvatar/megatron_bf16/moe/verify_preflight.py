"""Pure-CPU check of the native MoE availability evidence, never a speedup gate."""
import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(base):
    receipt, report, entry = (read(base / name) for name in ('receipt.json', 'moe.json', 'entry.json'))
    require(receipt['status'] == 'completed' and receipt['exit_code'] == 0 and receipt['performance_claims'] == [],
            'native preflight must complete without a performance claim')
    for name, digest in receipt['files'].items():
        require(Path(name).name == name and sha(base / name) == digest, 'original preflight input bytes differ')
    require(sha(base / 'launch_preflight.py') == receipt['wrapper_sha256'] == report['wrapper_sha256']
            and sha(base / 'run_preflight.py') == receipt['runner_sha256'], 'original observed wrapper/runner missing')
    manifest = read(HERE.parent.parent / 'megatron_training_entry/source_manifest.json')
    require(entry['status'] == 'completed' and entry['native_sources_sha256'] == manifest['files']
            and entry['megatron_commit'] == manifest['commit'] == '5be9626709af2722333bf54797c954c09edeada3'
            and entry['iluvatar_kernels'] is False, 'unchanged native source/candidate-free entry required')
    require(report['status'] == 'native_available' and report['candidate_enabled'] is False
            and report['performance_claims'] == [], 'this proves availability only')
    for name, digest in report['native_source_sha256'].items():
        require(manifest['files'][name] == digest, 'observed native MoE source differs')
    compat = report['bool_sort_compatibility']
    expected = {(length, mode, descending) for length in (0, 1, 17, 1024)
                for mode in ('zeros', 'ones', 'alternating', 'skewed') for descending in (False, True)}
    require(compat['enabled'] is True and compat['calls'] == 12
            and len(compat['probes']) == 32
            and {(r['length'], r['mode'], r['descending']) for r in compat['probes']} == expected
            and all(r['exact_indices_equal'] is True for r in compat['probes']),
            'explicit32-case stable-index mapping admission is required')
    calls = report['moe_calls']
    require(len(calls) == 12 and sorted(row['layer'] for row in calls) == [1]*3+[2]*3+[3]*3+[4]*3,
            'all four layers and three steps must execute')
    for row in calls:
        require(row['experts'].endswith('.SequentialMLP') and row['router'].endswith('.TopKRouter')
                and row['dispatcher'].endswith('.MoEAllGatherTokenDispatcher'), 'actual native backend differs')
        expected_config = {'num_moe_experts': 4, 'moe_router_topk': 2, 'moe_grouped_gemm': False,
            'moe_token_dispatcher_type': 'allgather', 'moe_apply_probs_on_input': False,
            'moe_aux_loss_coeff': .01, 'moe_z_loss_coeff': None, 'gated_linear_unit': True,
            'bias_activation_fusion': True, 'masked_softmax_fusion': True}
        require(row['config'] == expected_config, 'native router/activation/fusion configuration differs')
    require(len(report['expert_calls']) == len(report['probability_gradients']) == 12,
            'expert and router probability backward observations missing')
    for row in report['expert_calls']:
        counts = row['tokens_per_expert']
        require(len(counts) == 4 and all(type(count) is int and count >= 0 for count in counts)
                and sum(counts) == 512 and row['hidden_shape'] == [512, 512]
                and row['hidden_dtype'] == 'torch.bfloat16'
                and row['probability_shape'] == [512] and row['probability_requires_grad'] is True,
                'real dispatched token and probability shapes differ')
    require(all(r['finite'] is True and r['nonzero_elements'] > 0 and r['shape'] == [512]
                for r in report['probability_gradients']), 'finite nonzero probability gradient coverage missing')
    require(len(report['training_steps']) == 3, 'three actual native updates required')
    for row in report['training_steps']:
        require(row['parameters'] == 55652864 and row['parameter_tensors'] == row['main_grad_tensors']
                    == row['master_tensors'] == 56 and row['native_optimizer_classes'] == ['SGD']
                and row['parameter_dtypes'] == ['torch.bfloat16']
                and row['main_grad_dtypes'] == row['master_dtypes'] == ['torch.float32']
                and row['main_grads_finite'] is row['masters_finite'] is True,
                'native BF16/FP32-gradient/SGD-master observations differ')
    return {'status': 'availability_verified', 'receipt_sha256': sha(base / 'receipt.json'),
            'verifier_sha256': sha(Path(__file__)), 'native_sources': 600, 'training_steps': 3,
            'layer_calls': 12, 'stable_sort_probes': 32, 'parameters': 55652864,
            'expert_backend': 'SequentialMLP', 'candidate_correctness_compared': False,
            'performance_claims': [], 'gpu_rerun': False,
            'scope': 'CPU receipt/hash/config/dispatch/gradient-dtype checks of an explicit runtime-compatible '
                     'native preflight, not arithmetic replay, GroupedGEMM or accepted optimization.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=HERE / 'evidence/native-03')
    args = parser.parse_args()
    print(json.dumps(verify(args.base), indent=2))


if __name__ == '__main__':
    main()
