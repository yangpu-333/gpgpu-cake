"""Immutable three-step default-SwiGLU audit with both control and native gates.

This adds opt-in activation dispatch around the frozen BF16 launcher. The
original comparator retains every tensor, master-parameter, model and native
source gate. Its controlled comparison changes only the two checks that assume
the left arm has no CE candidate installed; independent native comparisons
retain those original checks as well.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


BASE = Path('/tmp/kda-swiglu-stage29')
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
SEED = 29101
STEPS = 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_case(folder, arm, candidate, contract, config, launcher):
    folder.mkdir(exist_ok=False)
    parent = BF / 'candidates/0008/candidate.py'
    command = [sys.executable, '-u', str(launcher),
               '--swiglu-receipt', str(folder / 'swiglu.json'),
               '--gradnorm-launcher', str(GRAD / 'launch_gradnorm-04.py'),
               '--bf16-root', str(BF),
               '--gradnorm-receipt', str(folder / 'gradnorm.json'),
               '--gradnorm-candidate', str(GRAD / 'candidate.py'),
               '--megatron-root', str(NATIVE), '--contract', str(contract),
               '--entry-evidence', str(folder / 'entry.json'),
               '--candidate', str(parent),
               '--audit-dir', str(folder / 'snapshots')]
    if arm == 'optimized':
        command += ['--swiglu-candidate', str(candidate)]
    command += config['training_argv'] + [
        '--train-iters', str(STEPS), '--seed', str(SEED), '--log-interval', '1',
        '--save', str(folder / 'snapshots'), '--save-interval', str(STEPS),
        '--save-wgrads-interval', '1', '--save-params-interval', '1']
    environment = os.environ.copy()
    for name in list(environment):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            environment.pop(name)
    environment.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1',
                       MASTER_PORT='29962', RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    record = {'case': folder.name, 'command': command, 'mode': 'optimized',
              'stage29_arm': arm, 'steps': STEPS, 'seed': SEED,
              'audit': True, 'profile': False, 'contract_sha256': sha(contract),
              'swiglu_candidate_sha256': sha(candidate) if arm == 'optimized' else None,
              'wrapper_sha256': sha(launcher), 'started_utc': utc()}
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
            process = subprocess.Popen(command, cwd=NATIVE, env=environment,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                record['exit_code'] = process.wait(timeout=240)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(exit_code=124, error='independent 240-second audit timeout')
        record['whole_process_elapsed_seconds'] = time.perf_counter() - started
    record['finished_utc'] = utc()
    write(folder / 'command.json', record)
    print(json.dumps({key: record[key] for key in
                      ('case', 'stage29_arm', 'exit_code', 'whole_process_elapsed_seconds')}),
          flush=True)
    require(record['exit_code'] == 0, arm + ' formal audit process failed')


def receipt_guard(control, optimized, candidate, config, launcher, manifest):
    left, right = read(control / 'swiglu.json'), read(optimized / 'swiglu.json')
    expected_source = {name: manifest['files'][name] for name in
                       ('megatron/core/transformer/mlp.py',
                        'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    expected_config = {'gated_linear_unit': True, 'bias_activation_fusion': True,
                       'activation_func': 'torch.nn.functional.silu',
                       'add_bias_linear': False, 'use_te_activation_func': False,
                       'activation_func_fp8_input_store': False, 'glu_linear_offset': 0.0,
                       'activation_func_clamp_value': None, 'cpu_offloading': False,
                       'per_token_scale_present': False}
    expected_calls = config['model']['num_layers'] * STEPS
    details = {'expected_swiglu_calls': expected_calls, 'candidate_sha256': sha(candidate),
               'wrapper_sha256': sha(launcher), 'control_receipt': left,
               'optimized_receipt': right, 'expected_configuration': expected_config}
    valid = (left['enabled'] is False and left['calls'] == 0
             and left['candidate_sha256'] is None and left['fallback_calls'] == 0
             and right['enabled'] is True and right['calls'] == expected_calls
             and right['native_calls'] == 0 and right['fallback_calls'] == 0
             and right['candidate_sha256'] == sha(candidate)
             and left['wrapper_sha256'] == right['wrapper_sha256'] == sha(launcher))
    for receipt in (left, right):
        valid = valid and receipt['native_source_sha256'] == expected_source
        valid = valid and receipt['native_compiled_callable'] is True
        valid = valid and receipt.get('configuration') == expected_config
    expected_shape = config['swiglu']['primary_shape']
    expected_stride = [expected_shape[1] * expected_shape[2], expected_shape[2], 1]
    shape_key = str((tuple(expected_shape), tuple(expected_stride), 'torch.bfloat16'))
    valid = valid and right['shapes'] == {shape_key: expected_calls}
    for folder in (control, optimized):
        grad = read(folder / 'gradnorm.json')
        valid = valid and (grad['enabled'] is True and grad['calls'] == STEPS
                           and grad['fallback_calls'] == 0
                           and grad['candidate_sha256'] == sha(GRAD / 'candidate.py')
                           and grad['wrapper_sha256'] == sha(GRAD / 'launch_gradnorm-04.py')
                           and len(grad['norm_audit']) == STEPS
                           and all(item['bitwise_equal'] for item in grad['norm_audit']))
    entry = read(control / 'entry.json')
    parent = BF / 'candidates/0008/candidate.py'
    valid = valid and (entry['iluvatar_kernels'] is True
                       and len(entry['kernel_adapters']) == 1
                       and entry['kernel_adapters'][0]['candidate_sha256'] == sha(parent)
                       and sha(parent) == config['candidate_sha256'])
    return valid, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='001')
    parser.add_argument('--case-suffix', default='complete')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--launcher', type=Path, default=BASE / 'launch_swiglu-02.py')
    args = parser.parse_args()
    require(args.version.isdigit() and args.case_suffix.replace('-', '').isalnum(),
            'simple immutable version and suffix required')
    suffix = args.version + '-' + args.case_suffix
    candidate = BASE / ('candidate-' + args.version + '.py')
    contract = BASE / 'contracts/main.json'
    config = read(contract)
    probe_path = args.probe or BASE / ('probe-' + args.version + '.json')
    probe = read(probe_path)
    require(probe['status'] == 'passed' and probe['candidate_sha256'] == sha(candidate),
            'independent candidate probe must pass and bind exact deployed candidate bytes')
    require(config['model']['parameters'] == 17895936
            and config['model']['num_layers'] == 4
            and config['model']['gated_linear_unit'] is True
            and config['model']['bias_activation_fusion'] is True
            and '--swiglu' in config['training_argv']
            and '--no-bias-swiglu-fusion' not in config['training_argv'],
            'sealed native-default SwiGLU main model required')
    native = BASE / 'main-native-audit'
    native_command = read(native / 'command.json')
    require(native_command['mode'] == 'native' and native_command['steps'] == STEPS
            and native_command['seed'] == SEED and native_command['exit_code'] == 0
            and native_command['audit'] is True and native_command['profile'] is False
            and native_command['contract_sha256'] == sha(contract),
            'successful same-contract original native audit is required')
    control = BASE / ('main-control-audit-' + suffix)
    optimized = BASE / ('main-optimized-audit-' + suffix)
    outputs = {
        'controlled': BASE / ('comparison-main-' + suffix + '.json'),
        'native_optimized': BASE / ('comparison-native-optimized-main-' + suffix + '.json'),
        'native_control': BASE / ('comparison-native-control-main-' + suffix + '.json')}
    summary_path = BASE / ('audit-summary-main-' + suffix + '.json')
    require(not any(path.exists() for path in
                    [control, optimized, summary_path, *outputs.values()]),
            'refusing to overwrite an immutable audit case or output')
    run_case(control, 'control', candidate, contract, config, args.launcher)
    run_case(optimized, 'optimized', candidate, contract, config, args.launcher)

    import torch
    spec = importlib.util.spec_from_file_location('frozen_bf16_comparison', BF / 'compare_training.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    manifest = read(BF.parent / 'megatron_training_entry/source_manifest.json')
    guard_passed, guard_details = receipt_guard(
        control, optimized, candidate, config, args.launcher, manifest)
    parent = BF / 'candidates/0008/candidate.py'

    class Controlled(frozen.Comparison):
        def require(self, condition, name, **details):
            if name == 'native has no candidate installation':
                condition = guard_passed
                name = 'Same CE0008 plus gradnorm001 control; only verified default SwiGLU candidate differs'
                details = guard_details
            elif name == 'native run mode':
                command = read(control / 'command.json')
                condition = command['mode'] == 'optimized' and command['stage29_arm'] == 'control'
                name = 'Controlled baseline explicitly opts into CE0008 and gradnorm001'
            return super().require(condition, name, **details)

    comparisons = {
        'controlled': Controlled(control, optimized, torch, contract, parent),
        'native_optimized': frozen.Comparison(native, optimized, torch, contract, parent),
        'native_control': frozen.Comparison(native, control, torch, contract, parent)}
    summary = {'schema': 'megatron-default-swiglu-three-way-audit-v1',
               'version': args.version, 'case_suffix': args.case_suffix,
               'candidate_sha256': sha(candidate), 'contract_sha256': sha(contract),
               'probe_path': str(probe_path), 'probe_sha256': sha(probe_path),
               'runner_sha256': sha(Path(__file__)),
               'frozen_comparator_sha256': sha(BF / 'compare_training.py'),
               'wrapper_sha256': sha(args.launcher), 'comparisons': {},
               'swiglu_receipt_guard_passed': guard_passed}
    for label, comparator in comparisons.items():
        report = comparator.run()
        report.update(stage29_comparison=label, swiglu_candidate_sha256=sha(candidate),
                      stage29_runner_sha256=summary['runner_sha256'],
                      swiglu_receipt_guard_passed=guard_passed)
        write(outputs[label], report)
        summary['comparisons'][label] = {
            'path': str(outputs[label]), 'sha256': sha(outputs[label]),
            'passed': report['passed'], 'check_summary': report['check_summary'],
            'tensor_comparisons': len(report['tensor_comparisons']),
            'all_tensors_bitwise_equal': all(item.get('exact_equal') is True
                                            for item in report['tensor_comparisons'])}
        print(json.dumps({'comparison': label, **summary['comparisons'][label]}), flush=True)
    summary.update(passed=guard_passed and all(item['passed'] for item in summary['comparisons'].values()),
                   finished_utc=utc(),
                   scope='Three untimed steps; fixed native default compiled SwiGLU. '
                         'All original model/source/tensor/master gates retained. '
                         'Passing permits a separate performance experiment; this is not throughput evidence.')
    write(summary_path, summary)
    print(json.dumps({'passed': summary['passed'], 'summary': str(summary_path)}), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
