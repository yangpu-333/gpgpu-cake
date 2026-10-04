"""Frozen stage31 contract, provenance, dispatch and original timing gates."""
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


BASE = Path('/tmp/kda-attention-stage31')
SEALED = Path('/tmp/kda-swiglu-stage29')
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
FORMAL = BF.parent / 'megatron_training_entry'
NATIVE = BF.parent / 'megatron-native-5be9626'
GRAD = Path('/tmp/kda-gradnorm-stage26')
WRAPPER = BASE / 'launch_attention.py'
DELEGATE = SEALED / 'launch_swiglu-03.py'
SWIGLU = SEALED / 'candidate-004.py'
SWIGLU_SHA = 'a516bb372012cd5569eed99db2815a41c0b229755158289ddf6536248d554250'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
SEEDS = [31102, 31103, 31104]
AUDIT_SEED = 31101
ORDERS = (('native', 'control', 'optimized'), ('optimized', 'control', 'native'),
          ('control', 'native', 'optimized'))
FROZEN = {
    'stage31/probe_full.py': '7778dc5837265b9fa4def2cf90d07ceeac2b992e583aa6939d74b5e99ad57fd0',
    'stage29/candidate-004.py': SWIGLU_SHA,
    'stage29/launch_swiglu-03.py': 'ad9e63f3361a284f2d137cd147c9dbf515b822e657d681a5de098724ffbddb1c',
    'stage29/contracts/main.json': 'b71917793c455625cb2a1e09733b6969f8e20b08058385408c3f5f7f930d723e',
    'stage29/probe-004-v3.json': 'afbb2c40c283cc16cfd42c5f27792068436c280a1de9959cb0d0e6bed7a0e342',
    'stage29/decision.json': 'd0dc77fcf3aded453f1e368d24e49ce2763b22683ca88d5fd22e8b90561e602f',
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
    'num_layers': 4, 'hidden_size': 512, 'num_attention_heads': 8, 'num_query_groups': 8,
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
    roots = {'stage31': BASE, 'stage29': SEALED, 'bf16': BF, 'formal': FORMAL, 'gradnorm': GRAD}
    require(group in roots and '..' not in Path(relative).parts, 'unsafe frozen source identity')
    return roots[group] / relative


def sources():
    return {name: sha(BASE / name) for name in
            ('common.py', 'launch_attention.py', 'run_audit.py', 'run_performance.py')}


def check_rows(rows):
    """Require every reported case and every contained numerical check to pass."""
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


def backend_gate(manifest):
    path = BASE / 'backend-default.json'
    backend = read(path)
    names = ('megatron/core/transformer/dot_product_attention.py',
             'megatron/core/fusions/fused_softmax.py', 'megatron/core/transformer/utils.py')
    expected = {name: manifest['files'][name] for name in names}
    require(backend['status'] == 'passed' and backend['requested_backend'] == 'default-fusion'
            and backend['sources_unchanged'] is True and backend['megatron_commit'] == COMMIT
            and backend['source_manifest_sha256'] == sha(FORMAL / 'source_manifest.json')
            and backend['controller_sha256'] == sha(BASE / 'probe_native_backend.py')
            and backend['native_source_sha256'] == expected,
            'original native default backend provenance differs')
    rows = backend['cases']
    require(len(rows) == 3 and {row['sequence'] for row in rows} == {128, 256, 512},
            'complete default native softmax shape feasibility evidence required')
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
                    'required default native CoreX extension was not loaded')
            current = {'path': item['path'], 'sha256': item['sha256']}
            require(Path(item['path']).is_file() and sha(item['path']) == item['sha256'],
                    'original default native extension binary differs')
            prior = extension_files.setdefault(name, current)
            require(prior == current, 'default backend probe changed binary between shapes')
    return {'path': str(path), 'sha256': sha(path), 'native_source_sha256': expected,
            'native_extensions': extension_files}


def prepare(version, probe_path=None):
    require(version.isdigit() and len(version) == 3, 'three-digit candidate version required')
    for name, expected in FROZEN.items():
        require(sha(frozen_path(name)) == expected, 'frozen dependency changed: ' + name)
    contract_path = BASE / 'contracts/main.json'
    contract, original = read(contract_path), read(SEALED / 'contracts/main.json')
    expected_argv = [arg for arg in original['training_argv'] if arg != '--no-masked-softmax-fusion']
    require(contract['training_argv'] == expected_argv and contract['model'] == original['model']
            and contract['correctness'] == original['correctness']
            and contract['candidate_sha256'] == original['candidate_sha256']
            and contract['model']['parameters'] == 17895936
            and contract['correctness']['steps'] == 3 and contract['correctness']['atol'] == 3e-4
            and contract['correctness']['rtol'] == 1e-3
            and contract['performance']['steps'] == 120
            and contract['performance']['independent_seeds'] == SEEDS
            and contract['performance']['arms'] == ['native', 'control', 'optimized']
            and contract['performance']['process_orders'] == [list(order) for order in ORDERS],
            'contract must retain frozen main model/precision/tolerances and enable default fusion in all arms')
    require(contract['stage31']['source_contract_sha256'] == sha(SEALED / 'contracts/main.json')
            and contract['stage31']['frozen_swiglu_candidate_sha256'] == SWIGLU_SHA
            and contract['attention']['shape'] == [128, 2, 8, 64]
            and contract['attention']['strides'] == [3072, 1536, 192, 1]
            and contract['attention']['storage_offsets'] == [0, 64, 128]
            and contract['attention']['scale'] == .125
            and contract['attention']['correctness']['audit_seed'] == AUDIT_SEED,
            'Attention scope/layout/parent contract differs')
    manifest = read(FORMAL / 'source_manifest.json')
    require(manifest['commit'] == contract['megatron_commit'] == COMMIT
            and len(manifest['files']) == 600, 'all600 fixed native sources required')
    backend = backend_gate(manifest)
    candidate = BASE / ('candidate-' + version + '.py')
    probe_path = Path(probe_path) if probe_path else BASE / ('probe-full-' + version + '.json')
    require(probe_path.resolve().parent == BASE.resolve(), 'probe must belong to stage31 root')
    probe = read(probe_path)
    require(probe['schema'] == 'native-fused-full-attention-independent-probe-v1'
            and probe['status'] == 'passed' and probe['version'] == version
            and probe['candidate_sha256'] == sha(candidate)
            and probe['native_source_sha256'] == backend['native_source_sha256']
            and probe['backend_proof_sha256'] == backend['sha256']
            and probe['atol'] == 3e-4 and probe['rtol'] == 1e-3,
            'passing full native Attention probe with unchanged tolerances and source binding required')
    check_rows(probe['checks'])
    check_rows(probe['guard_checks'])
    require(len(probe['checks']) >= 21
            and {tuple(row['shape']) for row in probe['checks']} >=
                {(128, 2, 8, 64), (256, 2, 8, 64), (512, 2, 8, 64)}
            and all(set(row['checks']) == {'output', 'query_grad', 'key_grad', 'value_grad'}
                    and row['inputs_unchanged'] is True for row in probe['checks'])
            and {row['case'] for row in probe['guard_checks']} >= {'cpu', 'float32'},
            'complete independent full Attention output/all-QKV-gradient/immutability/guard coverage required')
    replay = probe['native_capture_replay']
    require(set(replay) == {'output', 'query_grad', 'key_grad', 'value_grad'}
            and all(row['passed'] is True and row['exact_equal'] is True and row['bad_elements'] == 0
                    for row in replay.values()),
            'independent native reference must reproduce captured formal attention exactly')
    capture_folder = BASE / 'native-default-capture-02'
    capture = read(capture_folder / 'attention.json')
    require(capture['calls'] == 4 and capture['capture_written'] is True
            and capture['native_source_sha256'] == backend['native_source_sha256']
            and set(capture['raw_files']) == {'inputs.pt', 'output.pt', 'upstream-grad.pt',
                                             'query-grad.pt', 'key-grad.pt', 'value-grad.pt'},
            'fixed original native full Attention capture evidence differs')
    for name, item in capture['raw_files'].items():
        raw = capture_folder / 'attention' / name
        require(raw.stat().st_size == item['bytes'] and sha(raw) == item['sha256'],
                'captured native reference tensor bytes changed: ' + name)
    verifier = Path(probe.get('verifier_path', str(BASE / 'probe_full.py'))).resolve()
    require(verifier == (BASE / 'probe_full.py').resolve()
            and sha(verifier) == probe['verifier_sha256'] == FROZEN['stage31/probe_full.py'],
            'full independent Attention verifier source differs')
    swiglu_probe = read(SEALED / 'probe-004-v3.json')
    decision = read(SEALED / 'decision.json')
    require(decision['promoted'] is True and decision['selected_candidate'] == '004'
            and swiglu_probe['status'] == 'passed' and swiglu_probe['candidate_sha256'] == SWIGLU_SHA,
            'frozen SwiGLU004 control admission differs')
    provenance = {'frozen_sources_sha256': FROZEN, 'controller_sources_sha256': sources(),
                  'backend': backend, 'probe_path': str(probe_path), 'probe_sha256': sha(probe_path),
                  'verifier_path': str(verifier), 'verifier_sha256': sha(verifier),
                  'native_capture_receipt_sha256': sha(capture_folder / 'attention.json'),
                  'native_capture_raw_files': capture['raw_files'],
                  'candidate_sha256': sha(candidate), 'swiglu_candidate_sha256': SWIGLU_SHA,
                  'native_source_commit': COMMIT, 'native_source_file_count': 600,
                  'adaptation': 'Only outer launcher identity and controlled-left candidate identity gates '
                                'adapted. Original model/source/all-tensor/master/precision/timer gates retained.'}
    return contract_path, contract, manifest, provenance


def command_for(folder, arm, version, contract_path, contract, steps, seed):
    command = [sys.executable, '-u', str(WRAPPER), '--attention-receipt', str(folder / 'attention.json'),
               '--delegate-launcher', str(DELEGATE), '--swiglu-receipt', str(folder / 'swiglu.json'),
               '--gradnorm-launcher', str(GRAD / 'launch_gradnorm-04.py'),
               '--bf16-root', str(BF), '--gradnorm-receipt', str(folder / 'gradnorm.json'),
               '--megatron-root', str(NATIVE), '--contract', str(contract_path),
               '--entry-evidence', str(folder / 'entry.json')]
    if arm != 'native':
        command += ['--candidate', str(BF / 'candidates/0008/candidate.py'),
                    '--gradnorm-candidate', str(GRAD / 'candidate.py'), '--swiglu-candidate', str(SWIGLU)]
    if arm == 'optimized':
        command += ['--attention-candidate', str(BASE / ('candidate-' + version + '.py'))]
    if steps == 3:
        command += ['--audit-dir', str(folder / 'snapshots')]
    command += contract['training_argv'] + ['--train-iters', str(steps), '--seed', str(seed),
                                           '--log-interval', '1' if steps == 3 else '10']
    if steps == 3:
        command += ['--save', str(folder / 'snapshots'), '--save-interval', '3',
                    '--save-wgrads-interval', '1', '--save-params-interval', '1']
    return command


def run_case(folder, arm, version, contract_path, contract, steps, seed):
    folder.mkdir(exist_ok=False)
    command = command_for(folder, arm, version, contract_path, contract, steps, seed)
    environment = os.environ.copy()
    for name in list(environment):
        if 'API_KEY' in name or 'AUTH_TOKEN' in name:
            environment.pop(name)
    environment.update(CUDA_VISIBLE_DEVICES='0', MASTER_ADDR='127.0.0.1',
                       MASTER_PORT='29981' if steps == 3 else '29982',
                       RANK='0', WORLD_SIZE='1', LOCAL_RANK='0')
    record = {'case': folder.name, 'command': command, 'stage31_arm': arm,
              'mode': 'native' if arm == 'native' else 'optimized', 'steps': steps, 'seed': seed,
              'audit': steps == 3, 'profile': False, 'contract_sha256': sha(contract_path),
              'wrapper_sha256': sha(WRAPPER), 'controller_sources_sha256': sources(),
              'attention_candidate_sha256': sha(BASE / ('candidate-' + version + '.py')) if arm == 'optimized' else None,
              'swiglu_candidate_sha256': SWIGLU_SHA if arm != 'native' else None, 'started_utc': utc()}
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
                      ('case', 'stage31_arm', 'exit_code', 'whole_process_elapsed_seconds')}), flush=True)
    require(record['exit_code'] == 0, 'formal training process failed: ' + folder.name)
    return record


def hook_gate(folder, arm, version, contract, steps, manifest, backend):
    attention, activation, grad = (read(folder / name) for name in
                                    ('attention.json', 'swiglu.json', 'gradnorm.json'))
    use_attention, controlled = arm == 'optimized', arm != 'native'
    calls = contract['model']['num_layers'] * steps
    candidate_sha = sha(BASE / ('candidate-' + version + '.py'))
    observe_routes = steps == 3
    require(attention['schema'] == 'native-complete-attention-dispatch-v1'
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
            and attention['candidate_sha256'] == (candidate_sha if use_attention else None)
            and attention['native_source_sha256'] == backend['native_source_sha256']
            and attention['capture_written'] is False,
            'actual attention/source/zero fallback or audit-only native route observation scope differs')
    require(len(attention['configurations']) == 4
            and all(row == ATTENTION_CONFIG for row in attention['configurations']),
            'actual full Attention model/backend configuration differs')
    signatures = [(tuple([128, 2, 8, 64]), tuple([3072, 1536, 192, 1]), 'torch.bfloat16', offset, True, True)
                  for offset in (0, 64, 128)]
    signatures.append(((2, 1, 128, 128), (16384, 16384, 128, 1), 'torch.bool', 0, False, True))
    require(attention['shapes'] == {str(tuple(signatures)): calls},
            'actual QKV shape/strides/offsets/dtype/mask coverage differs')
    for name, expected in backend['native_extensions'].items():
        require(sha(expected['path']) == expected['sha256'], 'default native CoreX binary changed')
        if name == 'scaled_masked_softmax_cuda' or not use_attention:
            require(attention['native_extensions'].get(name) == expected,
                    'native formal attention extension differs from strong baseline proof')
    swiglu_sources = {name: manifest['files'][name] for name in
                      ('megatron/core/transformer/mlp.py',
                       'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
    require(activation['enabled'] is controlled and activation['calls'] == (calls if controlled else 0)
            and activation['fallback_calls'] == activation['native_calls'] == 0
            and activation['wrapper_sha256'] == sha(DELEGATE)
            and activation['candidate_sha256'] == (SWIGLU_SHA if controlled else None)
            and activation['native_source_sha256'] == swiglu_sources
            and activation['native_compiled_callable'] is True and activation.get('extension') is None
            and not activation.get('capture_written', False), 'frozen native or SwiGLU004 dispatch differs')
    shape_key = str(((128, 2, 4096), (8192, 4096, 1), 'torch.bfloat16'))
    require(activation['shapes'] == ({shape_key: calls} if controlled else {}),
            'same SwiGLU004 shape/call coverage differs')
    if controlled or steps == 3:
        require(activation.get('configuration') == SWIGLU_CONFIG, 'actual default SwiGLU configuration differs')
    require(grad['enabled'] is controlled and grad['calls'] == (steps if controlled else 0)
            and grad['fallback_calls'] == 0
            and grad['wrapper_sha256'] == sha(GRAD / 'launch_gradnorm-04.py')
            and grad['candidate_sha256'] == (sha(GRAD / 'candidate.py') if controlled else None),
            'same frozen gradnorm001 dispatch differs')
    require(len(grad['norm_audit']) == (3 if controlled and steps == 3 else 0),
            'gradient norm diagnostic instrumentation scope differs')
    require(all(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm']
                for row in grad['norm_audit']), 'actual gradient norm differs from native')
    if steps == 120:
        require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation', 'attention')),
                'tensor capture/audit/profile in timed case')
    return {'attention': attention, 'swiglu': activation, 'gradnorm': grad,
            'raw_sha256': {name: sha(folder / name) for name in
                           ('attention.json', 'swiglu.json', 'gradnorm.json')}}


def load_comparator():
    path = BF / 'compare_training.py'
    spec = importlib.util.spec_from_file_location('stage31_frozen_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_performance_validator():
    path = BF / 'summarize_performance.py'
    spec = importlib.util.spec_from_file_location('stage31_frozen_performance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = path.read_text()
    start = source.index("    invocation=command['command']")
    end = source.index("    runtime=entry['runtime_compatibility']", start)
    replacement = """    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact full Attention outer wrapper invocation')
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
                    'reason': 'Only exact direct-launch invocation replaced by the full Attention/SwiGLU/gradnorm '
                              'outer wrapper. All original model/source/runtime/precision/CE/timer gates retained.'}
