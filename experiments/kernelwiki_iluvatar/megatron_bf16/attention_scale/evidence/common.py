"""Independent stage32 held-out controls; frozen Attention003 and native math."""
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


BASE = Path('/tmp/kda-attention-scale-stage32')
ATTENTION = Path('/tmp/kda-attention-stage31')
SEALED = Path('/tmp/kda-swiglu-stage29')
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
FORMAL = BF.parent / 'megatron_training_entry'
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
WRAPPER = BASE / 'launch_attention_scale.py'
DELEGATE = GRAD / 'launch_gradnorm-04.py'
CANDIDATE = ATTENTION / 'candidate-003.py'
CANDIDATE_SHA = '4881f91c230324f258cf6faf7b1d63a493b043b960df4efdca0dad708195db48'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
MODEL_SIZES = {'wide': (4, 256, 4096, 21107200), 'deep': (8, 512, 8192, 42213888)}
SEEDS = {'wide': [32102, 32103, 32104], 'deep': [32202, 32203, 32204]}
AUDIT_SEEDS = {'wide': 32101, 'deep': 32201}
PROBE_TRIAL = '0032026100801'
CURRENT_BACKEND = BASE / 'backend-current-20261008-01.json'
ORDERS = (('native', 'control', 'optimized'), ('optimized', 'control', 'native'),
          ('control', 'native', 'optimized'))
FROZEN = {
    'stage31/candidate-003.py': CANDIDATE_SHA,
    'stage31/probe_full.py': '7778dc5837265b9fa4def2cf90d07ceeac2b992e583aa6939d74b5e99ad57fd0',
    'stage31/probe-full-003.json': '15144ae11bc7a4a8d052f1ec678195cf0d01f78396f9188772bc8f25c6be813f',
    'stage31/probe_native_backend.py': '8036e4a0218f7e5b5f2cedcaa16a34c4859e192203a71157b689f0183d3b87e1',
    'stage31/backend-default.json': 'fa399ad4038edbae7cd078c2d50bc588f4c91e5563f0a2bbce0d22c05af3f1df',
    'stage31/native-default-capture-02/attention.json': 'c4bf61c70b1b041dfbdadee41584285e741b64bad0910a223ea64613e59a3722',
    'stage31/audit-summary-main-003-complete.json': '7273c58aa1123b95da413bf11a0f3797669c7bf4fc1f13d227788483c9a8ad43',
    'stage31/performance-main-003-complete.json': 'ebcdd90fe307dce06d208410f1b02629c382c00f50bfb762c0cf89a8740d9958',
    'stage29/contracts/wide.json': '45d7a6e20198755cf54be0c6026dfa0ca22a979cd89863b47cd7a7e27bee8ccf',
    'stage29/contracts/deep.json': '4f980d7b3c9ed4ac5ecbc9be0b2e36f3b35c12efc574a2b42fbe76148829019a',
    'bf16/compare_training.py': 'feb475eb504fe452267f930f2069d5f17b36387a6cb4ad0f839a6aaa8df27eda',
    'bf16/summarize_performance.py': '6130e7cd9a2ffa26b42db42142694781bd45f9bdaa83336d4f1944a48368e5b7',
    'bf16/launch_training.py': 'a7909648600a4ef8448e738a75e5346845a392bdbcf2fbffc2fadecae1297e90',
    'bf16/kernel_adapter.py': 'bd84038d9037c8d62262a2a36d1cf378faa695bf2a9f3cd7288c02bc52037296',
    'bf16/run_case.py': '6f3fede9c52e872033ff086968d31cfa6d839cedf49fc8f1a4c86c71944ed6d3',
    'bf16/candidates/0008/candidate.py': '9b22e12b4744b2a41b9209faa280e361e6defe1d5480f5e5683fddd247b7f493',
    'formal/runtime_compat.py': '6a7e4bd6130ea4b2141d3ac2e3f00a284da2915d997a76249272322ef6973bff',
    'formal/source_manifest.json': 'a72ded960488d399ded2255ef55f357257e03532b8fa21a06d958170c5c98c2d',
    'formal/summarize_performance.py': '6c2d1f7955f7e474a3de0b9e4edc156bc9ba90b5f23c7a5fce696ee736dcd2d3',
    'gradnorm/candidate.py': '84c851ce5af7a49ba97685710790146d2c315f9c8837259366bffd7831f67dc0',
    'gradnorm/launch_gradnorm-04.py': '12503df82b69f7eb5059cc0c0722677e05ff14cdf1a37b6c34c0d7e3c8acba7f',
}
SWIGLU_CONFIG = {'gated_linear_unit': True, 'bias_activation_fusion': True,
                 'activation_func': 'torch.nn.functional.silu', 'add_bias_linear': False,
                 'use_te_activation_func': False, 'activation_func_fp8_input_store': False,
                 'glu_linear_offset': 0.0, 'activation_func_clamp_value': None,
                 'cpu_offloading': False, 'per_token_scale_present': False}
ATTENTION_CONFIG = {
    'hidden_size': 512, 'num_attention_heads': 8, 'num_query_groups': 8,
    'kv_channels': 64, 'attention_dropout': 0., 'masked_softmax_fusion': True,
    'attention_softmax_in_fp32': False, 'apply_query_key_layer_scaling': False,
    'sequence_parallel': False, 'context_parallel_size': 1, 'tensor_model_parallel_size': 1,
    'softmax_type': 'vanilla', 'window_size': None, 'bf16': True, 'fp16': False,
    'softmax_scale': .125, 'softmax_offset_present': False, 'attn_mask_type': 'AttnMaskType.causal',
    'dropout_probability': 0., 'tp_group_size': 1, 'heads_per_partition': 8,
    'groups_per_partition': 8, 'head_dimension': 64, 'hidden_per_partition': 512,
    'scaled_masked_softmax_fusion': True, 'softmax_in_fp32': False,
    'additional_softmax_scale': None, 'softmax_window_size': None,
    'softmax_attn_mask_type': 'AttnMaskType.causal', 'softmax_input_bf16': True,
    'softmax_input_fp16': False, 'native_kernel_available': True}
JIT_STAGES = ('_fwd_scores', '_fwd_context', '_bwd_context', '_bwd_scores')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def frozen_path(name):
    group, relative = name.split('/', 1)
    roots = {'stage31': ATTENTION, 'stage29': SEALED, 'bf16': BF, 'formal': FORMAL, 'gradnorm': GRAD}
    require(group in roots and '..' not in Path(relative).parts, 'unsafe frozen source identity')
    return roots[group] / relative


def sources():
    return {name: sha(BASE / name) for name in
            ('common.py', 'launch_attention_scale.py', 'run_audit.py', 'run_performance.py',
             'contracts/wide.json', 'contracts/deep.json')}


def check_rows(rows):
    require(isinstance(rows, list) and bool(rows), 'probe must contain nonempty per-case records')
    for row in rows:
        require(isinstance(row, dict) and row.get('passed') is True, 'probe case rejected')
        nested = row.get('checks')
        if nested is not None:
            values = list(nested.values()) if isinstance(nested, dict) else nested
            require(isinstance(values, list) and bool(values)
                    and all(isinstance(value, dict) and value.get('passed') is True
                            and value.get('bad_elements', 0) == 0 for value in values),
                    'probe numerical subcheck rejected or missing')


def backend_gate(manifest, path=None):
    path = Path(path) if path is not None else ATTENTION / 'backend-default.json'
    backend = read(path)
    expected = {name: manifest['files'][name] for name in
                ('megatron/core/transformer/dot_product_attention.py',
                 'megatron/core/fusions/fused_softmax.py', 'megatron/core/transformer/utils.py')}
    require(backend['status'] == 'passed' and backend['requested_backend'] == 'default-fusion'
            and backend['sources_unchanged'] is True and backend['megatron_commit'] == COMMIT
            and backend['source_manifest_sha256'] == sha(FORMAL / 'source_manifest.json')
            and backend['controller_sha256'] == sha(ATTENTION / 'probe_native_backend.py')
            and backend['native_source_sha256'] == expected, 'default native backend provenance differs')
    rows = backend['cases']
    require(len(rows) == 3 and {row['sequence'] for row in rows} == {128, 256, 512},
            'complete default native softmax feasibility evidence required')
    extension_files = {}
    for row in rows:
        require(row['status'] == 'passed' and row['route'] == 'cuda-extension'
                and all(row[key] is True for key in
                        ('kernel_available', 'forward_completed', 'backward_completed',
                         'cuda_extension_forward_completed', 'cuda_extension_backward_completed',
                         'scaled_masked_softmax_fusion', 'finite_output', 'finite_gradient', 'mask_unchanged'))
                and row['softmax_in_fp32'] is False and row['additional_softmax_scale'] is None
                and row['softmax_offset'] is None and row['input_dtype'] == 'torch.bfloat16',
                'default native extension forward/backward route was not proved')
        for name in ('scaled_masked_softmax_cuda', 'scaled_upper_triang_masked_softmax_cuda'):
            item = row['extensions_after'][name]
            require(item['loaded'] is True and item['importable'] is True,
                    'required native CoreX extension was not loaded')
            current = {'path': item['path'], 'sha256': item['sha256']}
            require(Path(item['path']).is_file() and sha(item['path']) == item['sha256'],
                    'original native extension binary differs')
            require(extension_files.setdefault(name, current) == current,
                    'default backend changed binary between shapes')
    return {'path': str(path), 'sha256': sha(path), 'native_source_sha256': expected,
            'native_extensions': extension_files}


def prepare(model, version='003', probe_path=None):
    require(model in MODEL_SIZES and version == '003', 'only frozen Attention003 wide/deep validation allowed')
    for name, expected in FROZEN.items():
        require(sha(frozen_path(name)) == expected, 'frozen dependency changed: ' + name)
    contract_path = BASE / ('contracts/' + model + '.json')
    contract, original = read(contract_path), read(SEALED / ('contracts/' + model + '.json'))
    expected_argv = [arg for arg in original['training_argv'] if arg != '--no-masked-softmax-fusion']
    require(contract['training_argv'] == expected_argv and contract['model'] == original['model']
            and contract['correctness'] == original['correctness']
            and contract['candidate_sha256'] == original['candidate_sha256'],
            'only original held-out model plus all-arm default masked-softmax fusion allowed')
    layers, sequence, vocabulary, parameters = MODEL_SIZES[model]
    cfg = contract['model']
    require(cfg['num_layers'] == layers and cfg['sequence_length'] == sequence
            and cfg['vocab_size'] == vocabulary and cfg['parameters'] == parameters
            and cfg['hidden_size'] == 512 and cfg['ffn_hidden_size'] == 2048
            and cfg['micro_batch_size'] == cfg['global_batch_size'] == 2
            and cfg['bf16'] is True and cfg['normalization'] == 'RMSNorm'
            and cfg['epsilon'] == 1e-5 and cfg['fp32_residual_connection'] is False
            and cfg['gated_linear_unit'] is True and cfg['bias_activation_fusion'] is True
            and contract['correctness']['atol'] == 3e-4 and contract['correctness']['rtol'] == 1e-3
            and contract['correctness']['steps'] == 3,
            'fixed larger model/precision/default activation configuration differs')
    require(contract['performance']['steps'] == 120
            and contract['performance']['independent_seeds'] == SEEDS[model]
            and contract['performance']['arms'] == ['native', 'control', 'optimized']
            and contract['performance']['process_orders'] == [list(order) for order in ORDERS]
            and '--swiglu' in expected_argv and '--no-bias-swiglu-fusion' not in expected_argv
            and '--no-masked-softmax-fusion' not in expected_argv,
            'native timer/rotation/default SwiGLU and softmax fusion contract differs')
    require(contract['stage32']['model_name'] == model
            and contract['stage32']['source_contract_sha256'] == sha(SEALED / ('contracts/' + model + '.json'))
            and contract['stage32']['attention_candidate_sha256'] == CANDIDATE_SHA
            and contract['stage32']['audit_seed'] == AUDIT_SEEDS[model]
            and contract['stage32']['candidate_version'] == version
            and contract['stage32']['probe_trial_label'] == PROBE_TRIAL
            and contract['stage32']['fresh_native_backend'] == CURRENT_BACKEND.name
            and contract['stage32']['fresh_probe'] == 'probe-full-' + PROBE_TRIAL + '.json'
            and contract['stage32']['independent_verifier_sha256'] == FROZEN['stage31/probe_full.py']
            and contract['stage32']['swiglu_candidate_enabled'] is False
            and contract['stage32']['rmsnorm_candidate_enabled'] is False
            and contract['stage32']['heldout_code_retuned'] is False
            and contract['attention']['shape'] == [sequence, 2, 8, 64]
            and contract['attention']['strides'] == [3072, 1536, 192, 1]
            and contract['attention']['storage_offsets'] == [0, 64, 128]
            and contract['attention']['scale'] == .125,
            'stage32 fixed candidate/scope/layout/parent contract differs')
    manifest = read(FORMAL / 'source_manifest.json')
    require(manifest['commit'] == contract['megatron_commit'] == COMMIT
            and len(manifest['files']) == 600, 'all600 fixed native sources required')
    require(sha(BF / 'candidates/0008/candidate.py') == contract['candidate_sha256'], 'CE0008 changed')
    historical_backend = backend_gate(manifest)
    backend = backend_gate(manifest, CURRENT_BACKEND)
    require(backend['native_source_sha256'] == historical_backend['native_source_sha256']
            and backend['native_extensions'] == historical_backend['native_extensions'],
            'fresh Pod backend must preserve the original source and CoreX binary identity')
    backend.update(historical_path=historical_backend['path'],
                   historical_sha256=historical_backend['sha256'])
    probe_path = Path(probe_path) if probe_path else BASE / ('probe-full-' + PROBE_TRIAL + '.json')
    require(probe_path.resolve() == (BASE / ('probe-full-' + PROBE_TRIAL + '.json')).resolve(),
            'use the fresh declared recheck of the frozen verifier; no candidate/version retuning')
    probe = read(probe_path)
    require(probe['schema'] == 'native-fused-full-attention-independent-probe-v1'
            and probe['status'] == 'passed' and probe['version'] == PROBE_TRIAL
            and probe['candidate_sha256'] == sha(CANDIDATE) == CANDIDATE_SHA
            and probe['native_source_sha256'] == backend['native_source_sha256']
            and probe['backend_proof_sha256'] == backend['historical_sha256']
            and probe['verifier_sha256'] == sha(ATTENTION / 'probe_full.py')
            and probe['atol'] == 3e-4 and probe['rtol'] == 1e-3
            and probe['runtime']['torch'] == '2.4.1'
            and probe['runtime']['selected_megatron_source'] == str(NATIVE)
            and probe['runtime']['optional_backends_disabled_in_process'] == ['transformer_engine', 'apex'],
            'fresh passing unchanged independent Attention probe/current runtime required')
    check_rows(probe['checks'])
    check_rows(probe['guard_checks'])
    require(len(probe['checks']) == 22
            and {tuple(row['shape']) for row in probe['checks']} >=
                {(128, 2, 8, 64), (256, 2, 8, 64), (512, 2, 8, 64)}
            and all(set(row['checks']) == {'output', 'query_grad', 'key_grad', 'value_grad'}
                    and row['inputs_unchanged'] is True for row in probe['checks'])
            and {row['case'] for row in probe['guard_checks']} >= {'cpu', 'float32'},
            'full output/all-QKV-gradient/immutability/guard coverage required')
    replay = probe['native_capture_replay']
    require(set(replay) == {'output', 'query_grad', 'key_grad', 'value_grad'}
            and all(row['passed'] is True and row['exact_equal'] is True and row['bad_elements'] == 0
                    for row in replay.values()), 'native full reference must reproduce frozen capture exactly')
    capture_folder = ATTENTION / 'native-default-capture-02'
    capture = read(capture_folder / 'attention.json')
    require(capture['calls'] == 4 and capture['capture_written'] is True
            and capture['native_source_sha256'] == backend['native_source_sha256']
            and set(capture['raw_files']) == {'inputs.pt', 'output.pt', 'upstream-grad.pt',
                                             'query-grad.pt', 'key-grad.pt', 'value-grad.pt'},
            'original native full Attention capture differs')
    for name, item in capture['raw_files'].items():
        raw = capture_folder / 'attention' / name
        require(raw.stat().st_size == item['bytes'] and sha(raw) == item['sha256'],
                'native capture tensor bytes changed: ' + name)
    admission_audit = read(ATTENTION / 'audit-summary-main-003-complete.json')
    admission_perf = read(ATTENTION / 'performance-main-003-complete.json')
    require(admission_audit['status'] == 'completed' and admission_audit['passed'] is True
            and admission_perf['status'] == 'validated' and admission_perf['promoted'] is True
            and admission_perf['candidate_sha256'] == CANDIDATE_SHA
            and admission_perf['audit_sha256'] == sha(ATTENTION / 'audit-summary-main-003-complete.json'),
            'previous main-model promotion admission differs; larger shapes remain fresh experiments')
    provenance = {'frozen_sources_sha256': FROZEN, 'controller_sources_sha256': sources(),
                  'backend': backend, 'probe_path': str(probe_path), 'probe_sha256': sha(probe_path),
                  'probe_trial_label': PROBE_TRIAL, 'candidate_version': version,
                  'historical_probe_sha256': sha(ATTENTION / 'probe-full-003.json'),
                  'verifier_path': str(ATTENTION / 'probe_full.py'),
                  'verifier_sha256': sha(ATTENTION / 'probe_full.py'),
                  'native_capture_receipt_sha256': sha(capture_folder / 'attention.json'),
                  'native_capture_raw_files': capture['raw_files'],
                  'candidate_path': str(CANDIDATE), 'candidate_sha256': CANDIDATE_SHA,
                  'swiglu_candidate_sha256': None, 'native_source_commit': COMMIT,
                  'native_source_file_count': 600,
                  'adaptation': 'Only declared larger model/seed/path/outer launcher identities and two '
                                'controlled-left candidate identity gates adapted. Original all-tensor, '
                                'master, source, precision, optimizer and timer gates retained. Default '
                                'native compiled SwiGLU and fused causal softmax in every arm; no Swi004.'}
    return contract_path, contract, manifest, provenance


def command_for(folder, arm, version, contract_path, contract, steps, seed):
    require(version == '003' and arm in ('native', 'control', 'optimized'), 'unsupported arm/version')
    command = [sys.executable, '-u', str(WRAPPER), '--attention-receipt', str(folder / 'attention.json'),
               '--swiglu-receipt', str(folder / 'swiglu.json'), '--delegate-launcher', str(DELEGATE),
               '--bf16-root', str(BF), '--gradnorm-receipt', str(folder / 'gradnorm.json'),
               '--megatron-root', str(NATIVE), '--contract', str(contract_path),
               '--entry-evidence', str(folder / 'entry.json')]
    if arm != 'native':
        command += ['--candidate', str(BF / 'candidates/0008/candidate.py'),
                    '--gradnorm-candidate', str(GRAD / 'candidate.py')]
    if arm == 'optimized':
        command += ['--attention-candidate', str(CANDIDATE)]
    if steps == 3:
        command += ['--audit-dir', str(folder / 'snapshots')]
    command += contract['training_argv'] + ['--train-iters', str(steps), '--seed', str(seed),
                                           '--log-interval', '1' if steps == 3 else '10']
    if steps == 3:
        command += ['--save', str(folder / 'snapshots'), '--save-interval', '3',
                    '--save-wgrads-interval', '1', '--save-params-interval', '1']
    return command


def run_case(folder, arm, model, version, contract_path, contract, steps, seed):
    folder.mkdir(exist_ok=False)
    command = command_for(folder, arm, version, contract_path, contract, steps, seed)
    environment = os.environ.copy()
    for name in list(environment):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            environment.pop(name)
    environment.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1',
                       MASTER_PORT='29991' if steps == 3 else '29992',
                       RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    record = {'case': folder.name, 'command': command, 'stage32_arm': arm, 'model_name': model,
              'mode': 'native' if arm == 'native' else 'optimized', 'steps': steps, 'seed': seed,
              'audit': steps == 3, 'profile': False, 'contract_sha256': sha(contract_path),
              'wrapper_sha256': sha(WRAPPER), 'controller_sources_sha256': sources(),
              'attention_candidate_sha256': CANDIDATE_SHA if arm == 'optimized' else None,
              'swiglu_candidate_sha256': None, 'started_utc': utc()}
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        started = time.perf_counter()
        with (folder / 'stdout.log').open('x') as out, (folder / 'stderr.log').open('x') as err:
            process = subprocess.Popen(command, cwd=NATIVE, env=environment,
                                       stdout=out, stderr=err, start_new_session=True)
            try:
                record['exit_code'] = process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(exit_code=124, error='independent 600-second held-out case timeout')
        record['whole_process_elapsed_seconds'] = time.perf_counter() - started
    record['finished_utc'] = utc()
    write(folder / 'command.json', record)
    print(json.dumps({key: record[key] for key in
                      ('case', 'stage32_arm', 'exit_code', 'whole_process_elapsed_seconds')}), flush=True)
    require(record['exit_code'] == 0, 'formal training process failed: ' + folder.name)
    return record


def hook_gate(folder, arm, version, contract, steps, manifest, backend):
    require(version == '003', 'candidate must remain frozen003')
    attention, activation, grad = (read(folder / name) for name in
                                    ('attention.json', 'swiglu.json', 'gradnorm.json'))
    use_attention, controlled, observe_routes = arm == 'optimized', arm != 'native', steps == 3
    layers, sequence = contract['model']['num_layers'], contract['model']['sequence_length']
    calls = layers * steps
    require(attention['schema'] == 'native-complete-attention-stage32-dispatch-v1'
            and attention['enabled'] is use_attention and attention['calls'] == calls
            and attention['candidate_calls'] == (calls if use_attention else 0)
            and attention['native_calls'] == (0 if use_attention else calls)
            and attention['fallback_calls'] == attention['unsupported_calls'] == 0
            and attention['native_routes_observed'] is observe_routes
            and attention['native_fused_forward_calls'] == (calls if observe_routes and not use_attention else 0)
            and attention['native_fused_backward_calls'] == (calls if observe_routes and not use_attention else 0)
            and attention['native_torch_forward_calls'] == 0
            and attention['wrapper_sha256'] == sha(WRAPPER)
            and attention['delegate_launcher_sha256'] == sha(DELEGATE)
            and attention['candidate_sha256'] == (CANDIDATE_SHA if use_attention else None)
            and attention['native_source_sha256'] == backend['native_source_sha256']
            and attention['capture_written'] is False,
            'actual Attention/source/fallback/audit-only observation scope differs')
    expected_config = dict(ATTENTION_CONFIG, num_layers=layers)
    require(len(attention['configurations']) == layers
            and all(row == expected_config for row in attention['configurations']),
            'actual full Attention model/default backend configuration differs')
    signatures = [(tuple([sequence, 2, 8, 64]), (3072, 1536, 192, 1), 'torch.bfloat16', offset, True, True)
                  for offset in (0, 64, 128)]
    signatures.append(((2, 1, sequence, sequence), (sequence ** 2, sequence ** 2, sequence, 1),
                       'torch.bool', 0, False, True))
    require(attention['shapes'] == {str(tuple(signatures)): calls}, 'QKV/mask layout and coverage differs')
    for name, expected in backend['native_extensions'].items():
        require(sha(expected['path']) == expected['sha256']
                and attention['native_extensions'].get(name) == expected,
                'actual native CoreX extension binary differs from strong baseline proof')
    jit = attention['candidate_jit']
    require(set(jit) == (set(JIT_STAGES) if use_attention else set())
            and all(row['script_function'] is True and isinstance(row['graph_sha256'], str)
                    and len(row['graph_sha256']) == 64 and row['graph_sha256'] == sha_text(row['graph'])
                    for row in jit.values()), 'candidate silently fell back from its four ScriptFunctions')
    if use_attention:
        cache = attention['candidate_backend_cache']
        require(len(cache) == 1 and cache[0]['shape'] == [sequence, sequence, 2, 8]
                and isinstance(cache[0]['batch_per_block'], int) and cache[0]['batch_per_block'] > 0
                and 16 % cache[0]['batch_per_block'] == 0
                and attention['candidate_scale_mode'] in ('float', 'tensor'),
                'frozen candidate fused availability/cache/scale ABI differs')
    else:
        require(attention['candidate_backend_cache'] == [] and attention['candidate_scale_mode'] is None,
                'native/control imported Attention candidate')
    swiglu_sources = {name: manifest['files'][name] for name in
                      ('megatron/core/transformer/mlp.py',
                       'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    require(activation['schema'] == 'native-default-swiglu-stage32-observation-v1'
            and activation['enabled'] is False and activation['candidate_sha256'] is None
            and activation['candidate_calls'] == activation['fallback_calls'] == 0
            and activation['native_calls_observed'] is observe_routes
            and activation['calls'] == activation['native_calls'] == (calls if observe_routes else 0)
            and activation['wrapper_sha256'] == sha(WRAPPER)
            and activation['native_source_sha256'] == swiglu_sources
            and activation['native_compiled_callable'] is True
            and activation.get('extension') is None and activation['capture_written'] is False,
            'all arms must retain actual original compiled SwiGLU with no candidate')
    swi_key = str(((sequence, 2, 4096), (8192, 4096, 1), 'torch.bfloat16'))
    require(activation['shapes'] == ({swi_key: calls} if observe_routes else {})
            and activation['configuration_observed'] is observe_routes
            and activation['configurations'] == ([SWIGLU_CONFIG] if observe_routes else []),
            'native SwiGLU audit provenance or timing instrumentation differs')
    require(grad['enabled'] is controlled and grad['calls'] == (steps if controlled else 0)
            and grad['fallback_calls'] == 0 and grad['wrapper_sha256'] == sha(DELEGATE)
            and grad['candidate_sha256'] == (sha(GRAD / 'candidate.py') if controlled else None),
            'same frozen gradnorm001 dispatch differs')
    require(len(grad['norm_audit']) == (3 if controlled and steps == 3 else 0)
            and all(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm']
                    for row in grad['norm_audit']), 'actual gradient norm differs from native')
    if steps == 120:
        require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation', 'attention')),
                'capture/audit/profile in timed case')
    return {'attention': attention, 'swiglu': activation, 'gradnorm': grad,
            'raw_sha256': {name: sha(folder / name) for name in
                           ('attention.json', 'swiglu.json', 'gradnorm.json')}}


def sha_text(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def load_comparator():
    path = BF / 'compare_training.py'
    spec = importlib.util.spec_from_file_location('stage32_frozen_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_performance_validator():
    path = BF / 'summarize_performance.py'
    spec = importlib.util.spec_from_file_location('stage32_frozen_performance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text()
    start = source.index("    invocation=command['command']")
    end = source.index("    runtime=entry['runtime_compatibility']", start)
    replacement = """    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact stage32 Attention/gradnorm outer invocation')
    require(entry['formal_entry']==EXPECTED_FORMAL_ENTRY,'original pinned native pretrain entry')
"""
    adapted = source[:start] + replacement + source[end:]
    exec(compile(adapted, str(path), 'exec'), module.__dict__)
    module.EXPECTED_COMMANDS = {}
    module.EXPECTED_FORMAL_ENTRY = str(NATIVE / 'pretrain_gpt.py')
    return module, {'frozen_validator_sha256': sha(path), 'adapted_validator_sha256': sha_text(adapted),
                    'replaced_original_argv_block_sha256': sha_text(source[start:end]),
                    'replacement_argv_block_sha256': sha_text(replacement),
                    'reason': 'Only exact direct-launch invocation replaced by stage32 Attention/gradnorm '
                              'outer wrapper. All original model/source/runtime/precision/CE/timer gates retained.'}
