"""Three-arm native-loop SwiGLU throughput after immutable numerical gates.

The frozen BF16 performance validator is reused in memory. Only its original
direct-launcher argv check is replaced by an exact check of this outer wrapper
and the original pretrain_gpt.py entry; all remaining model/source/runtime,
precision, CE coverage and synchronized native interval-timer gates are kept.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time


BASE = Path('/tmp/kda-swiglu-stage29')
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
STEPS = 120
ORDERS = (('native', 'control', 'optimized'),
          ('optimized', 'control', 'native'),
          ('control', 'native', 'optimized'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def numerical_gate(candidate, contract, launcher, probe_path, audit_path, version):
    probe, audit = read(probe_path), read(audit_path)
    require(probe['status'] == 'passed' and probe['candidate_sha256'] == sha(candidate),
            'independent passing numerical probe must bind deployed candidate bytes')
    require(audit['schema'] == 'megatron-default-swiglu-three-way-audit-v1'
            and audit['passed'] is True and audit['swiglu_receipt_guard_passed'] is True
            and audit['version'] == version and audit['candidate_sha256'] == sha(candidate)
            and audit['contract_sha256'] == sha(contract)
            and audit['wrapper_sha256'] == sha(launcher)
            and audit['probe_sha256'] == sha(probe_path)
            and audit['runner_sha256'] == sha(BASE / 'run_candidate_audit.py')
            and audit['frozen_comparator_sha256'] == sha(BF / 'compare_training.py'),
            'same-version three-way audit and its frozen numerical sources must remain intact')
    expected = {'controlled', 'native_optimized', 'native_control'}
    require(set(audit['comparisons']) == expected, 'all three numerical comparisons required')
    audited_extension = None
    for label, item in audit['comparisons'].items():
        path = Path(item['path'])
        report = read(path)
        require(path.resolve().parent == BASE.resolve() and item['passed'] is True
                and item['sha256'] == sha(path) and report['passed'] is True
                and report['swiglu_receipt_guard_passed'] is True
                and report['swiglu_candidate_sha256'] == sha(candidate)
                and report['stage29_comparison'] == label
                and report['stage29_runner_sha256'] == audit['runner_sha256']
                and report['comparator_sha256'] == audit['frozen_comparator_sha256']
                and report['steps'] == 3 and report['atol'] == 3e-4 and report['rtol'] == 1e-3
                and report['check_summary']['failed'] == 0
                and all(check['passed'] for check in report['checks']),
                'unaltered passing full numerical comparison required: ' + label)
        if label == 'controlled':
            guards = [check for check in report['checks']
                      if 'optimized_receipt' in check and 'control_receipt' in check]
            require(len(guards) == 1, 'exact controlled numerical receipt guard required')
            audited_extension = guards[0]['optimized_receipt'].get('extension')
            if audited_extension is not None:
                require(sha(Path(audited_extension['path'])) == audited_extension['sha256'],
                        'audited compiled extension artifact bytes changed')
    return audit, audited_extension


def load_validator():
    path = BF / 'summarize_performance.py'
    spec = importlib.util.spec_from_file_location('frozen_swiglu_performance_validator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text()
    start = source.index("    invocation=command['command']")
    end = source.index("    runtime=entry['runtime_compatibility']", start)
    replacement = """    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact three-arm default SwiGLU outer wrapper invocation')
    require(entry['formal_entry']==EXPECTED_FORMAL_ENTRY,'original pinned formal pretrain entry')
"""
    adapted = source[:start] + replacement + source[end:]
    exec(compile(adapted, str(path), 'exec'), module.__dict__)
    module.EXPECTED_COMMANDS = {}
    module.EXPECTED_FORMAL_ENTRY = str(NATIVE / 'pretrain_gpt.py')
    return module, {
        'frozen_validator_sha256': sha(path),
        'adapted_validator_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
        'replaced_original_argv_block_sha256': hashlib.sha256(source[start:end].encode()).hexdigest(),
        'replacement_argv_block_sha256': hashlib.sha256(replacement.encode()).hexdigest(),
        'adaptation_reason': 'The original validator accepts only direct launch_training.py argv. '
                             'This experiment adds an explicit process-local activation/gradnorm '
                             'wrapper; replace only that argv block with exact expected argv and '
                             'the same pinned pretrain_gpt.py path. All other frozen gates remain.'}


def command_for(folder, arm, seed, candidate, contract, config, launcher):
    command = [sys.executable, '-u', str(launcher),
               '--swiglu-receipt', str(folder / 'swiglu.json'),
               '--gradnorm-launcher', str(GRAD / 'launch_gradnorm-04.py'),
               '--bf16-root', str(BF), '--gradnorm-receipt', str(folder / 'gradnorm.json'),
               '--megatron-root', str(NATIVE), '--contract', str(contract),
               '--entry-evidence', str(folder / 'entry.json')]
    if arm != 'native':
        command += ['--candidate', str(BF / 'candidates/0008/candidate.py'),
                    '--gradnorm-candidate', str(GRAD / 'candidate.py')]
    if arm == 'optimized':
        command += ['--swiglu-candidate', str(candidate)]
    return command + config['training_argv'] + [
        '--train-iters', str(STEPS), '--seed', str(seed), '--log-interval', '10']


def run_case(folder, arm, seed, command, contract, launcher):
    folder.mkdir(exist_ok=False)
    environment = os.environ.copy()
    for name in list(environment):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            environment.pop(name)
    environment.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1',
                       MASTER_PORT='29963', RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    record = {'case': folder.name, 'command': command,
              'mode': 'native' if arm == 'native' else 'optimized', 'stage29_arm': arm,
              'steps': STEPS, 'seed': seed, 'audit': False, 'profile': False,
              'contract_sha256': sha(contract), 'wrapper_sha256': sha(launcher),
              'started_utc': utc()}
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
                record.update(exit_code=124, error='independent 240-second performance timeout')
        record['whole_process_elapsed_seconds'] = time.perf_counter() - started
    record['finished_utc'] = utc()
    write(folder / 'command.json', record)
    require(record['exit_code'] == 0, 'formal performance process failed: ' + folder.name)
    return record


def hook_gate(folder, arm, candidate, config, launcher, manifest, audited_extension):
    activation, grad = read(folder / 'swiglu.json'), read(folder / 'gradnorm.json')
    enabled = arm == 'optimized'
    require(activation['enabled'] is enabled
            and activation['calls'] == (config['model']['num_layers'] * STEPS if enabled else 0)
            and activation['fallback_calls'] == 0
            and activation['wrapper_sha256'] == sha(launcher)
            and activation['candidate_sha256'] == (sha(candidate) if enabled else None)
            and activation['native_calls'] == 0,
            'SwiGLU candidate coverage or provenance differs')
    require(activation.get('extension') == (audited_extension if enabled else None),
            'performance extension artifact differs from full numerical audit')
    if enabled and audited_extension is not None:
        require(sha(Path(audited_extension['path'])) == audited_extension['sha256'],
                'performance extension artifact bytes changed')
    expected_source = {name: manifest['files'][name] for name in
                       ('megatron/core/transformer/mlp.py',
                        'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    require(activation['native_source_sha256'] == expected_source
            and activation['native_compiled_callable'] is True
            and not activation.get('capture_written', False),
            'native default compiled backend source or noninstrumented scope differs')
    if enabled:
        expected_config = {'gated_linear_unit': True, 'bias_activation_fusion': True,
                           'activation_func': 'torch.nn.functional.silu',
                           'add_bias_linear': False, 'use_te_activation_func': False,
                           'activation_func_fp8_input_store': False, 'glu_linear_offset': 0.0,
                           'activation_func_clamp_value': None, 'cpu_offloading': False,
                           'per_token_scale_present': False}
        shape = config['swiglu']['primary_shape']
        strides = [shape[1] * shape[2], shape[2], 1]
        key = str((tuple(shape), tuple(strides), 'torch.bfloat16'))
        require(activation.get('configuration') == expected_config
                and activation['shapes'] == {key: config['model']['num_layers'] * STEPS},
                'actual default SwiGLU configuration/dtype/layout differs')
    else:
        require(activation['shapes'] == {}, 'native or control activation received candidate dispatch')
    use_grad = arm != 'native'
    require(grad['enabled'] is use_grad and grad['calls'] == (STEPS if use_grad else 0)
            and grad['fallback_calls'] == 0 and grad['norm_audit'] == []
            and grad['candidate_sha256'] == (sha(GRAD / 'candidate.py') if use_grad else None)
            and grad['wrapper_sha256'] == sha(GRAD / 'launch_gradnorm-04.py'),
            'gradnorm performance coverage, provenance or instrumentation differs')
    require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation')),
            'tensor snapshot or profiler artifacts in timed case')
    return {'swiglu': activation, 'gradnorm': grad,
            'raw_sha256': {name: sha(folder / name) for name in ('swiglu.json', 'gradnorm.json')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='001')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--launcher', type=Path, default=BASE / 'launch_swiglu-02.py')
    args = parser.parse_args()
    require(args.version.isdigit()
            and all(value.replace('-', '').isalnum() for value in
                    (args.audit_suffix, args.run_suffix)), 'simple immutable version and suffixes required')
    candidate = BASE / ('candidate-' + args.version + '.py')
    contract = BASE / 'contracts/main.json'
    config = read(contract)
    probe_path = args.probe or BASE / ('probe-' + args.version + '.json')
    audit_path = BASE / ('audit-summary-main-' + args.version + '-' + args.audit_suffix + '.json')
    _, audited_extension = numerical_gate(
        candidate, contract, args.launcher, probe_path, audit_path, args.version)
    seeds = config['performance']['independent_seeds']
    require(seeds == [26002, 26003, 26004] and config['performance']['steps'] == STEPS
            and config['model']['num_layers'] == 4 and config['model']['parameters'] == 17895936
            and config['model']['micro_batch_size'] == config['model']['global_batch_size'] == 2
            and config['model']['gated_linear_unit'] is True
            and config['model']['bias_activation_fusion'] is True
            and '--swiglu' in config['training_argv']
            and '--no-bias-swiglu-fusion' not in config['training_argv'],
            'fixed main model and existing contract performance seeds required')
    parent = BF / 'candidates/0008/candidate.py'
    require(sha(parent) == config['candidate_sha256'], 'sealed CE0008 parent differs')
    suffix = args.version + '-' + args.run_suffix
    output = BASE / ('performance-main-' + suffix + '.json')
    folders = {(seed, arm): BASE / ('main-' + arm + '-perf-' + suffix + '-' + str(seed))
               for seed in seeds for arm in ('native', 'control', 'optimized')}
    require(not any(path.exists() for path in [output, *folders.values()]),
            'refusing to overwrite an immutable performance case or output')
    validator, adaptation = load_validator()
    manifest = read(BF.parent / 'megatron_training_entry/source_manifest.json')
    report = {'schema': 'megatron-default-swiglu-three-arm-performance-v1',
              'status': 'running', 'promoted': False, 'version': args.version,
              'candidate_sha256': sha(candidate), 'contract_sha256': sha(contract),
              'audit_path': str(audit_path), 'audit_sha256': sha(audit_path),
              'probe_path': str(probe_path), 'probe_sha256': sha(probe_path),
              'runner_sha256': sha(Path(__file__)), 'wrapper_sha256': sha(args.launcher),
              'audited_extension': audited_extension,
              'gradnorm_candidate_sha256': sha(GRAD / 'candidate.py'),
              'ce_candidate_sha256': sha(parent), 'model': config['model'],
              'seeds': seeds, 'orders': ORDERS, 'pairs': [], 'completed_cases': [],
              'scope': 'Pinned native default compiled SwiGLU; all GEMMs and native data/DDP/SGD '
                       'unchanged. Native synchronized time.time interval logs, 120 steps per '
                       'process and first20 discarded; ten complete 10-step intervals retained. '
                       'No diagnostic tensor hooks, snapshotting, profile, checkpoint or eval '
                       'during timing. Separate three-step tensor audit precedes measurement.',
              'promotion_rule': {'gain_vs_control_strictly_greater_than': 0.01,
                                 'all_three_seeds_faster_vs_control': True},
              'validator_adaptation': adaptation, 'started_utc': utc()}
    try:
        for seed, order in zip(seeds, ORDERS):
            arms = {}
            for arm in order:
                folder = folders[(seed, arm)]
                command = command_for(folder, arm, seed, candidate, contract, config, args.launcher)
                validator.EXPECTED_COMMANDS[str(folder)] = command
                record = run_case(folder, arm, seed, command, contract, args.launcher)
                result = validator.validate_case(folder, record['mode'], seed, config,
                                                 sha(contract), sha(parent))
                hooks = hook_gate(folder, arm, candidate, config, args.launcher, manifest,
                                  audited_extension)
                result.update(stage29_arm=arm, hooks=hooks)
                require(tuple(row['iteration'] for row in result['intervals']
                              if row['iteration'] != 1) == tuple(range(10, 121, 10))
                        and sum(bool(row['retained']) for row in result['intervals']) == 10,
                        'native first20 discard and final100 step interval coverage differs')
                arms[arm] = result
                report['completed_cases'].append({'arm': arm, 'seed': seed, 'case': folder.name,
                                                  'mean_iteration_ms': result['mean_iteration_ms']})
                print(json.dumps(report['completed_cases'][-1]), flush=True)
            require(arms['native']['model'] == arms['control']['model'] == arms['optimized']['model'],
                    'three-arm actual model construction differs')
            for first, second in zip(order, order[1:]):
                require(datetime.fromisoformat(arms[first]['finished_utc'])
                        <= datetime.fromisoformat(arms[second]['started_utc']),
                        'three-arm runs overlap or contract rotation order differs')
            native_ms, control_ms, optimized_ms = (arms[name]['mean_iteration_ms']
                                                   for name in ('native', 'control', 'optimized'))
            report['pairs'].append({'seed': seed, 'order': order, 'arms': arms,
                                    'vs_native': native_ms / optimized_ms,
                                    'vs_control': control_ms / optimized_ms,
                                    'control_vs_native': native_ms / control_ms})
        for metric in ('vs_native', 'vs_control', 'control_vs_native'):
            ratio = math.exp(statistics.mean(math.log(pair[metric]) for pair in report['pairs']))
            report[metric] = {'ratio': ratio, 'gain_percent': 100 * (ratio - 1),
                              'all_seeds_faster': all(pair[metric] > 1 for pair in report['pairs'])}
        require(len(report['completed_cases']) == 9 and len(report['pairs']) == 3,
                'all nine serial performance processes required')
        report['promoted'] = (report['vs_control']['ratio'] > 1.01
                              and report['vs_control']['all_seeds_faster'])
        report.update(status='validated', decision='promoted' if report['promoted'] else 'not_promoted')
    except Exception as error:
        report.update(status='rejected', promoted=False, decision='not_promoted',
                      error_type=type(error).__name__, error=str(error))
    report['finished_utc'] = utc()
    write(output, report)
    print(json.dumps({key: report.get(key) for key in
                      ('status', 'decision', 'promoted', 'vs_native', 'vs_control', 'error')}
                     | {'output': str(output)}), flush=True)
    return 0 if report['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
