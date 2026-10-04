"""Shared immutable stage30 controls around frozen native BF16 training."""
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


BASE = Path('/tmp/kda-swiglu-scale-stage30')
SEALED = Path('/tmp/kda-swiglu-stage29')
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
FORMAL = BF.parent / 'megatron_training_entry'
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
CANDIDATE = SEALED / 'candidate-004.py'
WRAPPER = SEALED / 'launch_swiglu-03.py'
CANDIDATE_SHA = 'a516bb372012cd5569eed99db2815a41c0b229755158289ddf6536248d554250'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
ORDERS = (('native', 'control', 'optimized'), ('optimized', 'control', 'native'),
          ('control', 'native', 'optimized'))
MODEL_SIZES = {'wide': (4, 256, 4096, 21107200), 'deep': (8, 512, 8192, 42213888)}
SEEDS = {'wide': [26012, 26013, 26014], 'deep': [26022, 26023, 26024]}
AUDIT_SEEDS = {'wide': 30101, 'deep': 30201}
EXPECTED_CONFIG = {'gated_linear_unit': True, 'bias_activation_fusion': True,
                   'activation_func': 'torch.nn.functional.silu', 'add_bias_linear': False,
                   'use_te_activation_func': False, 'activation_func_fp8_input_store': False,
                   'glu_linear_offset': 0.0, 'activation_func_clamp_value': None,
                   'cpu_offloading': False, 'per_token_scale_present': False}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def frozen_path(name):
    group, relative = name.split('/', 1)
    roots = {'stage29': SEALED, 'bf16': BF, 'formal': FORMAL, 'gradnorm': GRAD}
    require(group in roots and '..' not in Path(relative).parts,
            'unknown or unsafe frozen source identity')
    return roots[group] / relative


def prepare(model):
    record = read(BASE / 'adaptation-record.json')
    require(record['schema'] == 'fixed-swiglu-stage30-adaptation-v1', 'adaptation record schema differs')
    for name, expected in record['controller_sources_sha256'].items():
        require(sha(BASE / name) == expected, 'stage30 controller/contract differs: ' + name)
    for name, expected in record['frozen_sources_sha256'].items():
        require(sha(frozen_path(name)) == expected, 'frozen dependency differs: ' + name)
    require(sha(CANDIDATE) == CANDIDATE_SHA, 'candidate004 bytes may not change during holdout validation')
    contract_path = BASE / ('contracts/' + model + '.json')
    contract, original = read(contract_path), read(SEALED / ('contracts/' + model + '.json'))
    require(contract['training_argv'] == original['training_argv']
            and contract['correctness'] == original['correctness']
            and contract['model'] == original['model']
            and contract['candidate_sha256'] == original['candidate_sha256'],
            'holdout changes original model/data/precision/tolerance contract')
    require(contract['stage30']['source_contract_sha256'] == sha(SEALED / ('contracts/' + model + '.json'))
            and contract['stage30']['swiglu_candidate_sha256'] == CANDIDATE_SHA
            and contract['stage30']['audit_seed'] == AUDIT_SEEDS[model],
            'stage30 contract provenance/candidate/audit seed differs')
    layers, sequence, vocabulary, parameters = MODEL_SIZES[model]
    cfg = contract['model']
    require(cfg['num_layers'] == layers and cfg['sequence_length'] == sequence
            and cfg['vocab_size'] == vocabulary and cfg['parameters'] == parameters
            and cfg['hidden_size'] == 512 and cfg['ffn_hidden_size'] == 2048
            and cfg['micro_batch_size'] == cfg['global_batch_size'] == 2
            and cfg['bf16'] is True and cfg['normalization'] == 'RMSNorm'
            and cfg['epsilon'] == 1e-5 and cfg['fp32_residual_connection'] is False
            and cfg['gated_linear_unit'] is True and cfg['bias_activation_fusion'] is True
            and contract['swiglu']['primary_shape'] == [sequence, 2, 4096]
            and contract['correctness']['atol'] == 3e-4
            and contract['correctness']['rtol'] == 1e-3
            and contract['correctness']['steps'] == 3,
            'fixed holdout model/precision/default activation configuration differs')
    require(contract['performance']['steps'] == 120
            and contract['performance']['independent_seeds'] == SEEDS[model]
            and contract['performance']['arms'] == ['native', 'control', 'optimized']
            and contract['performance']['process_orders'] == [list(order) for order in ORDERS]
            and '--swiglu' in contract['training_argv']
            and '--no-bias-swiglu-fusion' not in contract['training_argv'],
            'formal native timer/rotation/default fusion contract differs')
    manifest = read(FORMAL / 'source_manifest.json')
    require(manifest['commit'] == contract['megatron_commit'] == COMMIT
            and len(manifest['files']) == 600, 'fixed complete native source manifest required')
    require(sha(BF / 'candidates/0008/candidate.py') == contract['candidate_sha256'],
            'frozen CE0008 parent differs')
    main = read(SEALED / 'decision.json')
    probe = read(SEALED / 'probe-004-v3.json')
    require(main['promoted'] is True and main['selected_candidate'] == '004'
            and probe['status'] == 'passed' and probe['candidate_sha256'] == CANDIDATE_SHA
            and len(probe['checks']) == 37 and len(probe['guard_checks']) == 4
            and all(all(item['passed'] for item in row['checks']) for row in probe['checks'])
            and all(item['passed'] for item in probe['guard_checks']),
            'frozen candidate004 main-scope admission evidence differs')
    return contract_path, contract, manifest, record


def command_for(folder, arm, contract_path, contract, steps, seed):
    command = [sys.executable, '-u', str(WRAPPER), '--swiglu-receipt', str(folder / 'swiglu.json'),
               '--gradnorm-launcher', str(GRAD / 'launch_gradnorm-04.py'),
               '--bf16-root', str(BF), '--gradnorm-receipt', str(folder / 'gradnorm.json'),
               '--megatron-root', str(NATIVE), '--contract', str(contract_path),
               '--entry-evidence', str(folder / 'entry.json')]
    if arm != 'native':
        command += ['--candidate', str(BF / 'candidates/0008/candidate.py'),
                    '--gradnorm-candidate', str(GRAD / 'candidate.py')]
    if arm == 'optimized':
        command += ['--swiglu-candidate', str(CANDIDATE)]
    if steps == 3:
        command += ['--audit-dir', str(folder / 'snapshots')]
    command += contract['training_argv'] + ['--train-iters', str(steps), '--seed', str(seed),
                                           '--log-interval', '1' if steps == 3 else '10']
    if steps == 3:
        command += ['--save', str(folder / 'snapshots'), '--save-interval', '3',
                    '--save-wgrads-interval', '1', '--save-params-interval', '1']
    return command


def run_case(folder, arm, model, contract_path, contract, steps, seed):
    folder.mkdir(exist_ok=False)
    command = command_for(folder, arm, contract_path, contract, steps, seed)
    environment = os.environ.copy()
    for name in list(environment):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            environment.pop(name)
    environment.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1',
                       MASTER_PORT='29971' if steps == 3 else '29972',
                       RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    record = {'case': folder.name, 'command': command, 'stage30_arm': arm, 'model_name': model,
              'mode': 'native' if arm == 'native' else 'optimized', 'steps': steps, 'seed': seed,
              'audit': steps == 3, 'profile': False, 'contract_sha256': sha(contract_path),
              'wrapper_sha256': sha(WRAPPER), 'common_sha256': sha(Path(__file__)),
              'swiglu_candidate_sha256': CANDIDATE_SHA if arm == 'optimized' else None,
              'adaptation_record_sha256': sha(BASE / 'adaptation-record.json'), 'started_utc': utc()}
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
            process = subprocess.Popen(command, cwd=NATIVE, env=environment,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                record['exit_code'] = process.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(exit_code=124, error='independent 300-second case timeout')
        record['whole_process_elapsed_seconds'] = time.perf_counter() - started
    record['finished_utc'] = utc()
    write(folder / 'command.json', record)
    print(json.dumps({key: record[key] for key in
                      ('case', 'stage30_arm', 'exit_code', 'whole_process_elapsed_seconds')}), flush=True)
    require(record['exit_code'] == 0, 'formal training process failed: ' + folder.name)
    return record


def hook_gate(folder, arm, contract, steps, manifest):
    activation, grad = read(folder / 'swiglu.json'), read(folder / 'gradnorm.json')
    enabled, use_grad = arm == 'optimized', arm != 'native'
    calls = contract['model']['num_layers'] * steps
    expected_sources = {name: manifest['files'][name] for name in
                        ('megatron/core/transformer/mlp.py',
                         'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    require(activation['enabled'] is enabled and activation['calls'] == (calls if enabled else 0)
            and activation['fallback_calls'] == 0 and activation['native_calls'] == 0
            and activation['wrapper_sha256'] == sha(WRAPPER)
            and activation['candidate_sha256'] == (CANDIDATE_SHA if enabled else None)
            and activation['native_source_sha256'] == expected_sources
            and activation['native_compiled_callable'] is True
            and activation.get('extension') is None,
            'fixed activation candidate/source/default baseline/fallback scope differs')
    if enabled:
        shape = contract['swiglu']['primary_shape']
        stride = [shape[1] * shape[2], shape[2], 1]
        key = str((tuple(shape), tuple(stride), 'torch.bfloat16'))
        require(activation['shapes'] == {key: calls}, 'actual activation shape/dtype/layout coverage differs')
    else:
        require(activation['shapes'] == {}, 'native/control used candidate activation dispatch')
    if enabled or steps == 3:
        require(activation.get('configuration') == EXPECTED_CONFIG,
                'actual heldout model does not use default native SwiGLU configuration')
    require(not activation.get('capture_written', False), 'extra activation tensor capture is forbidden')
    require(grad['enabled'] is use_grad and grad['calls'] == (steps if use_grad else 0)
            and grad['fallback_calls'] == 0
            and grad['wrapper_sha256'] == sha(GRAD / 'launch_gradnorm-04.py')
            and grad['candidate_sha256'] == (sha(GRAD / 'candidate.py') if use_grad else None),
            'identical native or gradient-norm001 dispatch/provenance differs')
    require(len(grad['norm_audit']) == (3 if use_grad and steps == 3 else 0),
            'norm correctness instrumentation scope differs')
    for row in grad['norm_audit']:
        require(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm'],
                'actual heldout training gradient norm differs from native')
    if steps == 120:
        require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation')),
                'tensor snapshots or profiling in timed case')
    return {'swiglu': activation, 'gradnorm': grad,
            'raw_sha256': {name: sha(folder / name) for name in ('swiglu.json', 'gradnorm.json')}}


def load_comparator():
    path = BF / 'compare_training.py'
    spec = importlib.util.spec_from_file_location('stage30_frozen_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_performance_validator():
    path = BF / 'summarize_performance.py'
    spec = importlib.util.spec_from_file_location('stage30_frozen_performance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text()
    start = source.index("    invocation=command['command']")
    end = source.index("    runtime=entry['runtime_compatibility']", start)
    replacement = """    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact fixed-code SwiGLU holdout outer wrapper invocation')
    require(entry['formal_entry']==EXPECTED_FORMAL_ENTRY,'original pinned native pretrain entry')
"""
    adapted = source[:start] + replacement + source[end:]
    exec(compile(adapted, str(path), 'exec'), module.__dict__)
    module.EXPECTED_COMMANDS = {}
    module.EXPECTED_FORMAL_ENTRY = str(NATIVE / 'pretrain_gpt.py')
    return module, {'frozen_validator_sha256': sha(path),
                    'adapted_validator_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
                    'replaced_original_argv_block_sha256': hashlib.sha256(source[start:end].encode()).hexdigest(),
                    'replacement_argv_block_sha256': hashlib.sha256(replacement.encode()).hexdigest(),
                    'reason': 'Only direct-launch argv validation changes for the frozen activation/gradnorm '
                              'wrapper. Every original native source/runtime/model/precision/CE/timer gate remains.'}
