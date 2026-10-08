"""Independent CPU verification of immutable stage32 Attention003 receipts.

No Torch, Triton, candidate or experiment controller is imported. Numerical
receipt coverage and original interval logs are checked; available raw bytes
are hashed. Tensor arithmetic and remote native binary bytes are not replayed.
Missing backups stay explicit and are never promoted into complete evidence.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
from types import SimpleNamespace


SOURCE = Path(__file__).resolve().parent
BF = SOURCE.parent
FORMAL = BF.parent / 'megatron_training_entry'
SEALED = BF / 'evidence/swiglu-stage29'
ATTENTION = BF / 'attention'
ANCHOR = ATTENTION / 'evidence/round003'
REMOTE = '/tmp/kda-attention-scale-stage32'
REMOTE_ATTENTION = '/tmp/kda-attention-stage31'
REMOTE_SEALED = '/tmp/kda-swiglu-stage29'
REMOTE_BF = '/tmp/kda-bf16-continuation/megatron_bf16'
REMOTE_FORMAL = '/tmp/kda-bf16-continuation/megatron_training_entry'
REMOTE_NATIVE = '/tmp/kda-bf16-continuation/megatron-native-5be9626'
REMOTE_GRAD = '/tmp/kda-gradnorm-stage26'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
CANDIDATE_SHA = '4881f91c230324f258cf6faf7b1d63a493b043b960df4efdca0dad708195db48'
COMPARATOR_SHA = 'feb475eb504fe452267f930f2069d5f17b36387a6cb4ad0f839a6aaa8df27eda'
PARSER_SHA = '6c2d1f7955f7e474a3de0b9e4edc156bc9ba90b5f23c7a5fce696ee736dcd2d3'
ANCHOR_READER_SHA = '5ec635e25b8360cef6eaef7bd4db12a87b0c43919ad5f705dcb12c4db99f3adb'
MODELS = {'wide': (4, 256, 4096, 21107200, 32101, [32102, 32103, 32104], 1475, 637),
          'deep': (8, 512, 8192, 42213888, 32201, [32202, 32203, 32204], 2371, 1165)}
ORDERS = [['native', 'control', 'optimized'], ['optimized', 'control', 'native'],
          ['control', 'native', 'optimized']]
CONTROLLERS = ('common.py', 'launch_attention_scale.py', 'run_audit.py', 'run_performance.py',
               'contracts/wide.json', 'contracts/deep.json')
CONTROLLER_SHA = {
    'common.py': '02d3949c524ba451687bc6655bce09f664447456446b9f1c6eb44e55ed0abd0c',
    'launch_attention_scale.py': '1f528939a0da292910f264670c550a03f3eb7caa087209a8abb1d1046cb1fe2e',
    'run_audit.py': 'd22dfd86f48e40d1fa21c939dc888e90a111f9aa590fe9d330ab5f7a15f046d1',
    'run_performance.py': '99dd90aabd494a00f34f9e0ba3a0ef7c80920419cd099e4e7c2b3723b43e694e',
    'contracts/wide.json': 'd89816a795d1c9e7d1b46137186b34265b0256b9051d5d26466733bcd44c5b4f',
    'contracts/deep.json': 'a473489c036e86a9269b9d91b9e8c0223889fb464adf17ff1a14b4d3422d6162',
}
PROBE_TRIAL = '0032026100801'
CURRENT_BACKEND = 'backend-current-20261008-01.json'
FRESH_PROOFS = {
    CURRENT_BACKEND: '0aaa2ef96b5eebb52b691151567e5e44659b0fdf64d477137f5b4c42860890eb',
    'probe-full-' + PROBE_TRIAL + '.json': 'b8509fc036bea49240878986cff2eb723849717636ad5e6620a7492d4bb572cb',
}
JIT_STAGES = {'_fwd_scores', '_fwd_context', '_bwd_context', '_bwd_scores'}
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
    'bf16/compare_training.py': COMPARATOR_SHA,
    'bf16/summarize_performance.py': '6130e7cd9a2ffa26b42db42142694781bd45f9bdaa83336d4f1944a48368e5b7',
    'bf16/launch_training.py': 'a7909648600a4ef8448e738a75e5346845a392bdbcf2fbffc2fadecae1297e90',
    'bf16/kernel_adapter.py': 'bd84038d9037c8d62262a2a36d1cf378faa695bf2a9f3cd7288c02bc52037296',
    'bf16/run_case.py': '6f3fede9c52e872033ff086968d31cfa6d839cedf49fc8f1a4c86c71944ed6d3',
    'bf16/candidates/0008/candidate.py': '9b22e12b4744b2a41b9209faa280e361e6defe1d5480f5e5683fddd247b7f493',
    'formal/runtime_compat.py': '6a7e4bd6130ea4b2141d3ac2e3f00a284da2915d997a76249272322ef6973bff',
    'formal/source_manifest.json': 'a72ded960488d399ded2255ef55f357257e03532b8fa21a06d958170c5c98c2d',
    'formal/summarize_performance.py': PARSER_SHA,
    'gradnorm/candidate.py': '84c851ce5af7a49ba97685710790146d2c315f9c8837259366bffd7831f67dc0',
    'gradnorm/launch_gradnorm-04.py': '12503df82b69f7eb5059cc0c0722677e05ff14cdf1a37b6c34c0d7e3c8acba7f',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def close(left, right, message):
    require(type(left) in (int, float) and type(right) in (int, float)
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), message)


def relative(value):
    path = PurePosixPath(str(value))
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts),
            'unsafe archive path: ' + str(value))
    return path


def literal_constants(path):
    result = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if isinstance(node.value, ast.Dict):
            try:
                result[node.targets[0].id] = {ast.literal_eval(key): result[item.id]
                                              if isinstance(item, ast.Name) else ast.literal_eval(item)
                                              for key, item in zip(node.value.keys, node.value.values)}
            except (ValueError, TypeError, KeyError):
                pass
        else:
            try:
                result[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return result


def load_anchor_reader():
    path = ATTENTION / 'verify_receipts.py'
    require(sha(path) == ANCHOR_READER_SHA, 'frozen stage31 CPU reader changed')
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                 else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        require(all(name.split('.')[0] not in ('torch', 'triton', 'megatron', 'candidate') for name in names),
                'CPU reader must not import GPU/candidate modules')
    spec = importlib.util.spec_from_file_location('stage32_frozen_stage31_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Archive:
    def __init__(self, args):
        self.args, self.base, self.model, self.version = args, args.base.resolve(), args.model, args.version
        require(self.version == '003', 'stage32 accepts frozen003 only, never a failed round')
        (self.layers, self.sequence, self.vocabulary, self.parameters, self.audit_seed,
         self.seeds, self.check_count, self.tensor_count) = MODELS[self.model]
        self.contract_path = self.raw('contracts/' + self.model + '.json')
        self.contract = read(self.contract_path)
        self.manifest = read(FORMAL / 'source_manifest.json')
        self.constants = literal_constants(self.raw('common.py'))
        self.frozen_hashes = self.constants['FROZEN']
        self.controllers = {name: sha(self.raw(name)) for name in CONTROLLERS}
        self.candidate_sha = CANDIDATE_SHA
        self.inventory, self.snapshot_records, self.hashed = {}, {}, {}
        self.snapshot_present = self.snapshot_missing = 0
        self.backup_info = {'inventory_available': False, 'receipt_available': False,
                            'inventory_files': 0, 'files_available': 0, 'files_missing': 0}

    def raw(self, name):
        result = self.base.joinpath(*relative(name).parts)
        require(result.resolve().is_relative_to(self.base) and result.is_file(),
                'missing or escaped small evidence: ' + str(result))
        return result

    def hooks(self, folder, arm, steps):
        attention, activation, grad = (read(folder / name) for name in
                                        ('attention.json', 'swiglu.json', 'gradnorm.json'))
        use_attention, controlled, observed = arm == 'optimized', arm != 'native', steps == 3
        calls = self.layers * steps
        require(attention['schema'] == 'native-complete-attention-stage32-dispatch-v1'
                and attention['enabled'] is use_attention and attention['calls'] == calls
                and attention['candidate_calls'] == (calls if use_attention else 0)
                and attention['native_calls'] == (0 if use_attention else calls)
                and attention['fallback_calls'] == attention['unsupported_calls'] == 0
                and attention['native_routes_observed'] is observed
                and attention['native_fused_forward_calls'] == (calls if observed and not use_attention else 0)
                and attention['native_fused_backward_calls'] == (calls if observed and not use_attention else 0)
                and attention['native_torch_forward_calls'] == 0
                and attention['wrapper_sha256'] == self.controllers['launch_attention_scale.py']
                and attention['delegate_launcher_sha256'] == self.frozen_hashes['gradnorm/launch_gradnorm-04.py']
                and attention['candidate_sha256'] == (CANDIDATE_SHA if use_attention else None)
                and attention['native_source_sha256'] == self.backend['native_source_sha256']
                and attention['capture_written'] is False,
                'actual Attention/default backend/source/fallback/audit-only observation differs')
        require(len(attention['configurations']) == self.layers
                and all(row == self.expected_attention_config for row in attention['configurations']),
                'actual full Attention/default fused configuration differs')
        signatures = [(tuple([self.sequence, 2, 8, 64]), (3072, 1536, 192, 1),
                       'torch.bfloat16', offset, True, True) for offset in (0, 64, 128)]
        signatures.append(((2, 1, self.sequence, self.sequence),
                           (self.sequence ** 2, self.sequence ** 2, self.sequence, 1),
                           'torch.bool', 0, False, True))
        require(attention['shapes'] == {str(tuple(signatures)): calls}, 'QKV/mask layout/offset coverage differs')
        require(attention['native_extensions'] == self.backend['native_extensions'],
                'actual CoreX extension identities differ from strong default baseline proof')
        jit = attention['candidate_jit']
        require(set(jit) == (JIT_STAGES if use_attention else set())
                and all(row['script_function'] is True and isinstance(row['graph'], str)
                        and re.fullmatch(r'[0-9a-f]{64}', row['graph_sha256']) is not None
                        and row['graph_sha256'] == hashlib.sha256(row['graph'].encode()).hexdigest()
                        for row in jit.values()), 'candidate silently fell back from four ScriptFunctions')
        if use_attention:
            cache = attention['candidate_backend_cache']
            require(len(cache) == 1 and cache[0]['shape'] == [self.sequence, self.sequence, 2, 8]
                    and type(cache[0]['batch_per_block']) is int and cache[0]['batch_per_block'] > 0
                    and 16 % cache[0]['batch_per_block'] == 0
                    and attention['candidate_scale_mode'] in ('float', 'tensor'),
                    'frozen candidate backend cache/scale ABI differs')
        else:
            require(attention['candidate_backend_cache'] == [] and attention['candidate_scale_mode'] is None,
                    'native/control imported Attention candidate')
        swiglu_sources = {name: self.manifest['files'][name] for name in
                          ('megatron/core/transformer/mlp.py',
                           'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
        require(activation['schema'] == 'native-default-swiglu-stage32-observation-v1'
                and activation['enabled'] is False and activation['candidate_sha256'] is None
                and activation['candidate_calls'] == activation['fallback_calls'] == 0
                and activation['native_calls_observed'] is observed
                and activation['calls'] == activation['native_calls'] == (calls if observed else 0)
                and activation['wrapper_sha256'] == self.controllers['launch_attention_scale.py']
                and activation['native_source_sha256'] == swiglu_sources
                and activation['native_compiled_callable'] is True
                and activation.get('extension') is None and activation['capture_written'] is False,
                'all arms must retain original native compiled SwiGLU with no candidate')
        swi_key = str(((self.sequence, 2, 4096), (8192, 4096, 1), 'torch.bfloat16'))
        require(activation['shapes'] == ({swi_key: calls} if observed else {})
                and activation['configuration_observed'] is observed
                and activation['configurations'] == ([self.swiglu_config] if observed else []),
                'native SwiGLU provenance or timing instrumentation differs')
        require(grad['enabled'] is controlled and grad['calls'] == (steps if controlled else 0)
                and grad['fallback_calls'] == 0
                and grad['wrapper_sha256'] == self.frozen_hashes['gradnorm/launch_gradnorm-04.py']
                and grad['candidate_sha256'] == (self.frozen_hashes['gradnorm/candidate.py'] if controlled else None),
                'frozen gradnorm001 source/dispatch differs')
        require(len(grad['norm_audit']) == (3 if controlled and observed else 0)
                and all(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm']
                        for row in grad['norm_audit']), 'actual gradient norm differs from native')
        if steps == 120:
            require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation', 'attention')),
                    'capture/audit/profile in timed case')
        return {'attention': attention, 'swiglu': activation, 'gradnorm': grad,
                'raw_sha256': {name: sha(folder / name) for name in
                               ('attention.json', 'swiglu.json', 'gradnorm.json')}}

    def entry(self, folder, arm, steps, seed):
        command, entry = read(folder / 'command.json'), read(folder / 'entry.json')
        require(command['case'] == folder.name and command['stage32_arm'] == arm
                and command['model_name'] == self.model and command['steps'] == steps and command['seed'] == seed and command['exit_code'] == 0
                and command['audit'] is (steps == 3) and command['profile'] is False
                and command['mode'] == ('native' if arm == 'native' else 'optimized')
                and command['contract_sha256'] == entry['contract_sha256'] == sha(self.contract_path)
                and command['controller_sources_sha256'] == self.controllers
                and command['wrapper_sha256'] == self.controllers['launch_attention_scale.py']
                and command['attention_candidate_sha256'] == (self.candidate_sha if arm == 'optimized' else None)
                and command['swiglu_candidate_sha256'] is None
                and entry['status'] == 'completed', 'successful process headers differ')
        require(command['command'][0] == '/root/miniconda3/bin/python'
                and command['command'] == self.expected_command(folder, arm, steps, seed, command['command'][0]),
                'exact original formal-entry outer invocation differs')
        integration = {name: sha(BF / name) for name in ('launch_training.py', 'kernel_adapter.py', 'run_case.py')}
        integration.update({'formal/' + name: sha(FORMAL / name) for name in ('runtime_compat.py', 'source_manifest.json')})
        require(entry['native_sources_sha256'] == self.manifest['files'] and entry['megatron_commit'] == COMMIT
                and entry['integration_sha256'] == integration and entry['launcher_sha256'] == integration['launch_training.py']
                and entry['formal_entry'] == REMOTE_NATIVE + '/pretrain_gpt.py'
                and entry['training_argv'] == self.contract['training_argv'] + self.training_suffix(folder, steps, seed)
                and entry['profile_hooks'] == [] and 'profiler_activities' not in entry,
                'all600 original sources/integration/training arguments differ')
        runtime = entry['runtime_compatibility']
        require(runtime['torch'] == '2.4.1' and runtime['optional_backends_disabled_in_process'] == ['transformer_engine', 'apex'],
                'same runtime compatibility required')
        expected = {'parameters': self.parameters, 'layers': self.layers, 'hidden_size': 512, 'vocab_size': self.vocabulary,
                    'bf16': True, 'fp16': False, 'class': 'megatron.core.models.gpt.gpt_model.GPTModel'}
        require(len(entry['models']) == 1 and all(entry['models'][0].get(key) == value for key, value in expected.items()),
                'actual built model differs')
        names = entry['models'][0]['parameter_names']
        state_keys = entry['models'][0]['state_dict_keys']
        extra_keys = {'output_layer._extra_state'}
        extra_keys.update('decoder.layers.' + str(layer) + '.' + block + '._extra_state'
                          for layer in range(self.layers) for block in
                          ('mlp.linear_fc1', 'mlp.linear_fc2', 'self_attention.linear_proj',
                           'self_attention.linear_qkv'))
        require(len(names) == len(set(names)) == 6 * self.layers + 4
                and len(state_keys) == len(set(state_keys)) and set(state_keys) == set(names) | extra_keys,
                'complete real native parameter-name/state-key coverage required')
        require(entry['iluvatar_kernels'] is (arm != 'native'), 'CE opt-in differs')
        if arm == 'native':
            require(entry['kernel_adapters'] == [], 'native installed CE candidate')
        else:
            require(len(entry['kernel_adapters']) == 1, 'exactly one CE adapter required')
            handle = entry['kernel_adapters'][0]
            require(all(handle[key] is True for key in ('enabled', 'active', 'provenance_verified'))
                    and handle['candidate_sha256'] == handle['candidate_expected_sha256'] == self.contract['candidate_sha256']
                    and handle['runtime_world_size'] == 1
                    and handle['verified_devices'] in ({'0': 'BI-V150'}, {'0': 'Iluvatar BI-V150'})
                    and handle['counts'] == {'ce_candidate_calls': steps, 'ce_native_calls': 0,
                                             'residual_native_calls': self.layers * steps, 'fallback_reasons': {}},
                    'CE source/device/coverage differs')
        require(datetime.fromisoformat(command['started_utc']) < datetime.fromisoformat(command['finished_utc'])
                and math.isfinite(command['whole_process_elapsed_seconds']) and command['whole_process_elapsed_seconds'] > 0,
                'invalid process chronology')
        return command, entry

    def tensors(self, proof):
        checks, tensors = proof['checks'], proof['tensor_comparisons']
        counts = {'total': len(checks), 'passed': sum(row['passed'] is True for row in checks),
                  'failed': sum(row['passed'] is not True for row in checks)}
        require(proof['check_summary'] == counts and counts['total'] == self.check_count
                and counts['failed'] == 0 and proof['passed'] is True
                and len(tensors) == self.tensor_count,
                'complete frozen gate/tensor record counts differ')
        require(len({(row['group'], row['path']) for row in tensors}) == len(tensors),
                'duplicate tensor comparisons cannot stand in for coverage')
        groups = {}
        for row in tensors:
            groups[row['group']] = groups.get(row['group'], 0) + 1
            require(row['passed'] is True and row['shape_dtype_layout_match'] is True
                    and row['all_elements_finite'] is True
                    and row['native_shape'] == row['optimized_shape']
                    and row['native_dtype'] == row['optimized_dtype']
                    and row['native_numel'] == row['optimized_numel']
                    and row['native_numel'] == math.prod(row['native_shape'])
                    and 0 <= row['elements_not_exactly_equal'] <= row['native_numel']
                    and row['elements_outside_tolerance'] == 0
                    and math.isfinite(row['max_tolerance_fraction'])
                    and 0 <= row['max_tolerance_fraction'] <= 1
                    and row['exact_equal'] is (row['elements_not_exactly_equal'] == 0)
                    and (row['native_sha256'] != row['optimized_sha256'] or row['exact_equal'] is True),
                    'tensor numerical/header fields contradict pass: ' + row['path'])
            exact_groups = {'initial', 'actual_start_model_parameters', 'actual_start_master_parameters',
                            'master_initialization', 'model_initialization', 'batches', 'master_model_consistency'}
            require(row['exact_required'] is (row['group'] in exact_groups),
                    'original strict initialization/input/master-model consistency scope changed')
            dtypes = {'actual_start_model_parameters':'torch.bfloat16',
                      'actual_start_master_parameters':'torch.float32',
                      'wgrads':'torch.float32','params':'torch.bfloat16','masters':'torch.float32',
                      'master_model_consistency':'torch.bfloat16'}
            if row['group'] in dtypes:
                require(row['native_dtype'] == dtypes[row['group']], 'BF16 model / FP32 gradients or masters changed')
            require(not row['exact_required'] or row['exact_equal'] is True,
                    'required exact initialization/input/master check differs')
            require(row['atol'] == (0 if row['exact_required'] else 3e-4)
                    and row['rtol'] == (0 if row['exact_required'] else 1e-3), 'tensor tolerance changed')
            for key in ('max_absolute_error', 'max_relative_error_to_clamped_native'):
                require(math.isfinite(row[key]) and row[key] >= 0, 'invalid tensor error metric')
            require(not row['exact_equal'] or row['max_absolute_error'] == 0,
                    'exact-equal tensor has nonzero absolute error')
        n = 6 * self.layers + 4
        expected_groups = {'initial': n, 'actual_start_model_parameters': n,
                           'actual_start_master_parameters': n, 'master_initialization': 2 * n,
                           'model_initialization': 2 * n, 'batches': 15, 'output': 3, 'derived_loss': 3,
                           'wgrads': 3 * n, 'params': 3 * n, 'masters': 3 * n,
                           'master_model_consistency': 6 * n}
        require(groups == proof['tensor_groups'] == expected_groups, 'all-parameter tensor coverage differs')
        require(bool(proof['raw_value_comparisons'])
                and all(row['passed'] is True for row in proof['raw_value_comparisons']),
                'recursive raw metadata comparisons failed or missing')
        expected_inputs = {REMOTE + '/contracts/' + self.model + '.json',
                           REMOTE_FORMAL + '/source_manifest.json', REMOTE_BF + '/candidates/0008/candidate.py'}
        for case in (proof['native_case'], proof['optimized_case']):
            expected_inputs.update(case + '/' + name for name in ('command.json', 'entry.json'))
            expected_inputs.update(case + '/snapshots/' + name for name in ('model0-initial.pt', 'actual-start.pt'))
            for step in range(1, 4):
                expected_inputs.update(case + '/snapshots/' + name for name in
                                       (f'model0-forward-{step:03d}.pt', f'master-after-{step:03d}.pt',
                                        f'wgrads/iter_{step:07d}/mp_rank_00.pth',
                                        f'params/iter_{step:07d}/mp_rank_00.pth'))
        require(len(proof['input_files']) == len(expected_inputs) == 35
                and {row['path'] for row in proof['input_files']} == expected_inputs,
                'all original contract/metadata/initial/input/gradient/parameter/master snapshot inputs required')
        for row in proof['input_files']:
            require(type(row['bytes']) is int and row['bytes'] >= 0
                    and re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None,
                    'invalid original input byte/hash receipt')
            mapped = self.recorded(row['path'])
            if row['path'].startswith(REMOTE + '/') and '/snapshots/' in row['path']:
                rel = row['path'][len(REMOTE) + 1:]
                previous = self.snapshot_records.setdefault(rel, row)
                require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                        'same snapshot has conflicting audit receipts')
                if self.inventory:
                    require(rel in self.inventory and self.inventory[rel]['bytes'] == row['bytes']
                            and self.inventory[rel]['sha256'] == row['sha256'],
                            'raw backup inventory disagrees with original numerical input')
                mapped = self.raw_snapshot(rel)
                if mapped.is_file():
                    self.byte_check(mapped, row)
            else:
                require(mapped.is_file(), 'missing inventoried small evidence: ' + str(mapped))
                self.byte_check(mapped, row)
        for group, multiplier in {'initial': 1, 'actual_start_model_parameters': 1,
                                  'actual_start_master_parameters': 1, 'master_initialization': 2,
                                  'model_initialization': 2, 'wgrads': 3, 'params': 3,
                                  'masters': 3, 'master_model_consistency': 6}.items():
            require(sum(row['native_numel'] for row in tensors if row['group'] == group)
                    == multiplier * self.parameters, 'full native parameter elements not covered: ' + group)
        require(all(row['native_shape'] == [2, self.sequence] and row['native_dtype'] == 'torch.float32'
                    for row in tensors if row['group'] == 'output'), 'per-token native model loss shape/dtype differs')
        exact = all(row['exact_equal'] is True for row in tensors)
        bitwise = all(row['native_sha256'] == row['optimized_sha256'] for row in tensors)
        return {'checks': counts['total'], 'tensors': len(tensors), 'tensor_groups': groups,
                'raw_value_comparisons': len(proof['raw_value_comparisons']),
                'all_tensors_exact_equal': exact, 'all_tensor_bytes_equal': bitwise}

    def audit(self):
        suffix = self.args.audit_suffix
        path = self.raw('audit-summary-' + self.model + '-' + self.version + '-' + suffix + '.json')
        audit = read(path)
        require(audit['schema'] == 'native-full-attention-stage32-three-way-audit-v1'
                and audit['status'] == 'completed' and audit['passed'] is True
                and audit['version'] == self.version and audit['model_name'] == self.model and audit['suffix'] == suffix
                and audit['audit_seed'] == self.audit_seed and audit['candidate_sha256'] == self.candidate_sha
                and audit['contract_sha256'] == sha(self.contract_path)
                and audit['wrapper_sha256'] == self.controllers['launch_attention_scale.py']
                and audit['runner_sha256'] == self.controllers['run_audit.py']
                and audit['controller_sources_sha256'] == self.controllers and audit['provenance'] == self.provenance
                and audit['frozen_comparator_sha256'] == sha(BF / 'compare_training.py'),
                'same-version full numerical audit scope/source/probe differs')
        require(set(audit['comparisons']) == {'controlled', 'native_optimized', 'native_control'}
                and [row['arm'] for row in audit['completed_cases']] == ['native', 'control', 'optimized'],
                'complete ordered three-arm numerical audit required')
        folders, commands, routes = {}, {}, {}
        for item in audit['completed_cases']:
            arm = item['arm']
            folder = self.recorded(item['path'])
            require(folder.parent == self.base and folder.name == self.model + '-' + arm + '-audit-' + self.version + '-' + suffix,
                    'unexpected numerical case root/name')
            command, entry = self.entry(folder, arm, 3, self.audit_seed)
            hooks = self.hooks(folder, arm, 3)
            require(item['hooks'] == hooks, 'raw numerical hook receipts differ from audit summary')
            folders[arm], commands[arm] = folder, command
            observation = hooks['attention']
            routes[arm] = {'case': folder.name, 'attention_receipt_sha256': hooks['raw_sha256']['attention.json'],
                           **{name: observation[name] for name in
                              ('native_routes_observed', 'native_fused_forward_calls', 'native_fused_backward_calls',
                               'native_torch_forward_calls', 'candidate_calls', 'fallback_calls')},
                           'native_swiglu_receipt_sha256': hooks['raw_sha256']['swiglu.json'],
                           'native_swiglu_calls': hooks['swiglu']['native_calls'],
                           'native_swiglu_configurations': hooks['swiglu']['configurations'],
                           'native_swiglu_compiled_callable': hooks['swiglu']['native_compiled_callable']}
        for first, second in zip(('native', 'control'), ('control', 'optimized')):
            require(datetime.fromisoformat(commands[first]['finished_utc']) <= datetime.fromisoformat(commands[second]['started_utc']),
                    'numerical processes overlap or order differs')
        results, inputs = {}, {}
        for label, (left, right) in {'controlled': ('control', 'optimized'),
                                   'native_optimized': ('native', 'optimized'), 'native_control': ('native', 'control')}.items():
            item = audit['comparisons'][label]
            report_path = self.recorded(item['path'])
            require(report_path.parent == self.base and sha(report_path) == item['sha256'], 'raw comparison hash differs')
            proof = read(report_path)
            require(proof['stage32_comparison'] == label and proof['model_name'] == self.model and proof['version'] == self.version
                    and proof['native_case'] == REMOTE + '/' + folders[left].name
                    and proof['optimized_case'] == REMOTE + '/' + folders[right].name
                    and proof['steps'] == 3 and proof['atol'] == 3e-4 and proof['rtol'] == 1e-3
                    and proof['comparison_device'] == 'CPU float64 copies'
                    and proof['comparator_sha256'] == audit['frozen_comparator_sha256']
                    and proof['stage32_runner_sha256'] == self.controllers['run_audit.py']
                    and proof['stage32_controller_sources_sha256'] == self.controllers
                    and proof['stage32_probe_sha256'] == self.probe_info['sha256']
                    and proof['attention_candidate_sha256'] == self.candidate_sha, 'comparison metadata/source differs')
            results[label] = self.tensors(proof)
            require(item['passed'] is True and item['check_summary'] == proof['check_summary']
                    and item['tensor_comparisons'] == results[label]['tensors']
                    and item['all_tensors_exact_equal'] is results[label]['all_tensors_exact_equal']
                    and item['all_tensor_bytes_equal'] is results[label]['all_tensor_bytes_equal'],
                    'raw numerical records contradict comparison summary')
            for row in proof['input_files']:
                previous = inputs.setdefault(row['path'], row)
                require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                        'same raw input has conflicting bytes/SHA receipts')
        self.input_count = len(inputs)
        self.input_rows = inputs
        self.snapshot_present = sum(self.raw_snapshot(rel).is_file() for rel in self.snapshot_records)
        self.snapshot_missing = len(self.snapshot_records) - self.snapshot_present
        return path, results, routes

    def load_validator(self):
        parser_path, path = FORMAL / 'summarize_performance.py', BF / 'summarize_performance.py'
        require(sha(parser_path) == PARSER_SHA, 'original standard-library native interval parser differs')
        for current in (parser_path, path):
            for node in ast.walk(ast.parse(current.read_text(encoding='utf-8'))):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                require(all(name.split('.')[0] not in ('torch', 'triton', 'megatron', 'candidate') for name in names),
                        'unexpected GPU/candidate dependency in CPU frozen reader')
        spec = importlib.util.spec_from_file_location('stage31_native_interval_parser', parser_path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        tree = ast.parse(self.raw('common.py').read_text(encoding='utf-8'))
        replacements = [ast.literal_eval(node.value) for function in tree.body
                        if isinstance(function, ast.FunctionDef) and function.name == 'load_performance_validator'
                        for node in function.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == 'replacement' for target in node.targets)]
        require(len(replacements) == 1, 'one explicit frozen outer-argv block adaptation required')
        source, replacement = path.read_text(encoding='utf-8'), replacements[0]
        start = source.index("    invocation=command['command']")
        end = source.index("    runtime=entry['runtime_compatibility']", start)
        adapted = source[:start] + replacement + source[end:]
        hashes = {'frozen_validator_sha256': sha(path),
                  'adapted_validator_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
                  'replaced_original_argv_block_sha256': hashlib.sha256(source[start:end].encode()).hexdigest(),
                  'replacement_argv_block_sha256': hashlib.sha256(replacement.encode()).hexdigest()}
        spec = importlib.util.spec_from_file_location('stage31_frozen_performance_validator', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        exec(compile(adapted, str(path), 'exec'), module.__dict__)
        module.parse_native_intervals = parser.parse_intervals
        module.EXPECTED_COMMANDS = {}
        module.EXPECTED_FORMAL_ENTRY = REMOTE_NATIVE + '/pretrain_gpt.py'
        return parser, module, hashes

    def performance(self, audit_path, audit_routes):
        path = self.base / ('performance-' + self.model + '-003-' + self.args.run_suffix + '.json')
        if not path.exists():
            return {'available': False}
        perf = read(path)
        require(self.recorded(perf['audit_path']).resolve() == audit_path.resolve()
                and perf['audit_sha256'] == sha(audit_path),
                'performance must bind the exact fresh complete numerical summary')
        require(perf['schema'] == 'native-full-attention-stage32-three-arm-performance-v1'
                and perf['status'] == 'validated' and perf['version'] == self.version
                and perf['model_name'] == self.model and perf['run_suffix'] == self.args.run_suffix
                and perf['candidate_sha256'] == self.candidate_sha and perf['contract_sha256'] == sha(self.contract_path)
                and perf['wrapper_sha256'] == self.controllers['launch_attention_scale.py']
                and perf['runner_sha256'] == self.controllers['run_performance.py']
                and perf['controller_sources_sha256'] == self.controllers and perf['provenance'] == self.provenance
                and perf['ce_candidate_sha256'] == self.contract['candidate_sha256']
                and perf['gradnorm_candidate_sha256'] == sha(self.frozen('gradnorm/candidate.py'))
                and perf['swiglu_candidate_sha256'] is None and perf['model'] == self.contract['model']
                and perf['performance_native_routes_observed_per_call'] is False
                and perf['performance_swiglu_observed_per_call'] is False
                and perf['untimed_audit_route_provenance'] == audit_routes
                and perf['numerical_input_files_hashed_before_timing'] == self.input_count,
                'performance source/probe/model/audit binding or route-observation fairness differs')
        parser, validator, hashes = self.load_validator()
        require(all(perf['validator_adaptation'][key] == value for key, value in hashes.items()),
                'original in-memory outer-argv-only adaptation differs')
        require(perf['seeds'] == self.seeds and perf['orders'] == ORDERS
                and len(perf['pairs']) == 3 and len(perf['completed_cases']) == 9, 'complete declared nine-process rotation required')
        ratios = {name: [] for name in ('vs_native', 'vs_control', 'control_vs_native')}
        completed, previous_finish = [], None
        for pair, seed, order in zip(perf['pairs'], self.seeds, ORDERS):
            require(pair['seed'] == seed and pair['order'] == order and set(pair['arms']) == {'native', 'control', 'optimized'},
                    'per-seed three-arm rotation differs')
            means, models = {}, {}
            for arm in order:
                folder = self.base / (self.model + '-' + arm + '-perf-' + self.version + '-' + self.args.run_suffix + '-' + str(seed))
                command, entry = self.entry(folder, arm, 120, seed)
                hooks = self.hooks(folder, arm, 120)
                validator.EXPECTED_COMMANDS[str(folder)] = self.expected_command(folder, arm, 120, seed, command['command'][0])
                fresh = validator.validate_case(folder, command['mode'], seed, self.contract,
                                                sha(self.contract_path), self.contract['candidate_sha256'])
                fresh.update(stage32_arm=arm, hooks=hooks)
                recorded = pair['arms'][arm]
                require(fresh == recorded, 'summary timing/model/hooks/raw hashes differ from frozen revalidation')
                intervals = parser.parse_intervals((folder / 'stdout.log').read_text(encoding='utf-8'))
                require(intervals == recorded['intervals']
                        and [row['iteration'] for row in intervals if row['iteration'] != 1] == list(range(10, 121, 10))
                        and [row['iteration'] for row in intervals if row['retained']] == list(range(30, 121, 10)),
                        'original 120-step/first20 exclusion/full100 retained intervals differ')
                means[arm] = statistics.mean(row['mean_iteration_ms'] for row in intervals if row['retained'])
                close(recorded['mean_iteration_ms'], means[arm], 'raw interval mean differs')
                started, finished = (datetime.fromisoformat(command[name]) for name in ('started_utc', 'finished_utc'))
                require(previous_finish is None or previous_finish <= started, 'timed processes overlap or rotation differs')
                previous_finish = finished
                models[arm] = entry['models']
                completed.append({'arm': arm, 'seed': seed, 'case': folder.name, 'mean_iteration_ms': means[arm]})
            require(models['native'] == models['control'] == models['optimized'], 'timed model metadata differs')
            values = {'vs_native': means['native'] / means['optimized'], 'vs_control': means['control'] / means['optimized'],
                      'control_vs_native': means['native'] / means['control']}
            for metric, value in values.items():
                close(pair[metric], value, 'per-seed ratio differs: ' + metric)
                ratios[metric].append(value)
        require(perf['completed_cases'] == completed, 'nine-case completed summary differs from raw cases')
        results = {}
        for metric, values in ratios.items():
            ratio = math.exp(statistics.mean(math.log(value) for value in values))
            results[metric] = {'ratio': ratio, 'gain_percent': 100 * (ratio - 1), 'all_seeds_faster': all(value > 1 for value in values)}
            close(perf[metric]['ratio'], ratio, 'geometric mean differs: ' + metric)
            close(perf[metric]['gain_percent'], results[metric]['gain_percent'], 'reported gain differs: ' + metric)
            require(perf[metric]['all_seeds_faster'] is results[metric]['all_seeds_faster'], 'all-seed positive flag differs')
        promoted = results['vs_control']['ratio'] > 1.01 and results['vs_control']['all_seeds_faster']
        require(perf['promoted'] is promoted and perf['decision'] == ('promoted_tested_model_only' if promoted else 'not_promoted')
                and perf['promotion_rule'] == {'gain_vs_control_strictly_greater_than': 0.01, 'all_three_seeds_faster_vs_control': True},
                'promotion contradicts strict>1% and all-three-seed same-round control gate')
        return {'available': True, 'sha256': sha(path), 'timed_processes': 9, 'run_steps': 1080, 'retained_steps': 900,
                'native_route_observed_per_timed_call': False, 'swiglu_observed_per_timed_call': False,
                'promoted': promoted,
                'audit_metadata_binding_passed': True,
                'diagnostic_only': False, **results}


    def recorded(self, value):
        value = str(value).replace('\\', '/')
        for remote, local in ((REMOTE, self.base), (REMOTE_ATTENTION, ANCHOR),
                              (REMOTE_SEALED, SEALED), (REMOTE_BF, BF), (REMOTE_FORMAL, FORMAL)):
            if value.startswith(remote + '/'):
                result = local.joinpath(*relative(value[len(remote) + 1:]).parts)
                require(result.resolve().is_relative_to(local.resolve()), 'recorded path escaped root')
                return result
        if value == REMOTE_GRAD + '/candidate.py':
            return BF / 'gradnorm_candidates/001/candidate.py'
        if value == REMOTE_GRAD + '/launch_gradnorm-04.py':
            return BF / 'evidence/gradnorm-stage26/launch_gradnorm-04.py'
        raise ValueError('unmapped recorded path: ' + value)

    def frozen(self, name):
        group, rest = name.split('/', 1)
        roots = {'stage31': REMOTE_ATTENTION, 'stage29': REMOTE_SEALED, 'bf16': REMOTE_BF,
                 'formal': REMOTE_FORMAL, 'gradnorm': REMOTE_GRAD}
        require(group in roots, 'unmapped frozen dependency')
        return self.recorded(roots[group] + '/' + relative(rest).as_posix())

    def byte_check(self, path, row):
        stat = path.stat()
        require(stat.st_size == row['bytes'], 'raw byte size differs: ' + str(path))
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in self.hashed:
            self.hashed[key] = sha(path)
        require(self.hashed[key] == row['sha256'], 'raw bytes differ: ' + str(path))

    def raw_snapshot(self, value):
        rel = relative(value)
        choices = [self.base.joinpath(*rel.parts), (self.base / 'raw-backup').joinpath(*rel.parts)]
        return next((path for path in choices if path.is_file()), choices[0])

    def foundations(self):
        require(self.frozen_hashes == FROZEN and len(FROZEN) == 21
                and all(re.fullmatch(r'[0-9a-f]{64}', value) for value in FROZEN.values()),
                'exact independent21-source frozen admission required')
        for name, digest in self.frozen_hashes.items():
            require(sha(self.frozen(name)) == digest, 'frozen source changed: ' + name)
        require(self.controllers == CONTROLLER_SHA, 'independently pinned six stage32 controllers/contracts changed')
        for name, digest in self.controllers.items():
            require(sha(SOURCE / name) == digest, 'project/deployed controller or contract differs: ' + name)
        require(self.constants['MODEL_SIZES'] == {name: value[:4] for name, value in MODELS.items()}
                and self.constants['SEEDS'] == {name: value[5] for name, value in MODELS.items()}
                and self.constants['AUDIT_SEEDS'] == {name: value[4] for name, value in MODELS.items()}
                and [list(order) for order in self.constants['ORDERS']] == ORDERS,
                'independent model/seed/rotation scope differs')
        original_path = SEALED / ('contracts/' + self.model + '.json')
        original = read(original_path)
        require(self.contract['model'] == original['model'] and self.contract['correctness'] == original['correctness']
                and self.contract['candidate_sha256'] == original['candidate_sha256']
                and self.contract['training_argv'] == [arg for arg in original['training_argv']
                                                     if arg != '--no-masked-softmax-fusion'],
                'only declared held-out model plus default fusion may replace the historical contract')
        cfg = self.contract['model']
        expected = {'num_layers': self.layers, 'sequence_length': self.sequence, 'vocab_size': self.vocabulary,
                    'parameters': self.parameters, 'hidden_size': 512, 'ffn_hidden_size': 2048,
                    'num_attention_heads': 8, 'micro_batch_size': 2, 'global_batch_size': 2, 'bf16': True,
                    'normalization': 'RMSNorm', 'epsilon': 1e-5, 'fp32_residual_connection': False,
                    'gated_linear_unit': True, 'bias_activation_fusion': True}
        require(all(cfg.get(key) == value for key, value in expected.items())
                and self.contract['correctness']['steps'] == 3
                and self.contract['correctness']['atol'] == 3e-4 and self.contract['correctness']['rtol'] == 1e-3,
                'real fixed model/precision/tolerance differs')
        stage = self.contract['stage32']
        require(stage['model_name'] == self.model and stage['source_contract_sha256'] == sha(original_path)
                and stage['attention_candidate_sha256'] == CANDIDATE_SHA and stage['audit_seed'] == self.audit_seed
                and stage['candidate_version'] == self.version and stage['probe_trial_label'] == PROBE_TRIAL
                and stage['fresh_native_backend'] == CURRENT_BACKEND
                and stage['fresh_probe'] == 'probe-full-' + PROBE_TRIAL + '.json'
                and stage['independent_verifier_sha256'] == FROZEN['stage31/probe_full.py']
                and stage['swiglu_candidate_enabled'] is stage['rmsnorm_candidate_enabled'] is stage['heldout_code_retuned'] is False,
                'candidate/parent contract/no-retuning/no-Swi004 scope differs')
        scope = self.contract['attention']
        require(scope['shape'] == [self.sequence, 2, 8, 64] and scope['strides'] == [3072, 1536, 192, 1]
                and scope['storage_offsets'] == [0, 64, 128] and scope['scale'] == .125,
                'real frozen candidate layout differs')
        perf = self.contract['performance']
        require(perf['steps'] == 120 and perf['independent_seeds'] == self.seeds
                and perf['arms'] == ['native', 'control', 'optimized'] and perf['process_orders'] == ORDERS
                and perf['tensor_snapshotting'] is False and '--swiglu' in self.contract['training_argv']
                and '--no-bias-swiglu-fusion' not in self.contract['training_argv']
                and '--no-masked-softmax-fusion' not in self.contract['training_argv'],
                'formal timer/rotation/default native fusion differs')
        require(self.manifest['commit'] == self.contract['megatron_commit'] == COMMIT
                and len(self.manifest['files']) == 600, 'all600 native sources required')
        reader = load_anchor_reader()
        anchor_args = SimpleNamespace(base=ANCHOR, version='003', probe=None,
                                      audit_suffix='complete', run_suffix='complete')
        anchor = reader.Archive(anchor_args)
        anchor.foundations()
        anchor.backup()
        anchor_audit, anchor_comparisons, anchor_routes = anchor.audit()
        anchor_perf = anchor.performance(anchor_audit, anchor_routes)
        require(anchor_perf['available'] is True and anchor_perf['promoted'] is True
                and anchor_perf['audit_metadata_binding_passed'] is True
                and anchor_perf['diagnostic_only'] is False, 'frozen003 admission must be a valid complete promotion')
        self.generation_info = anchor.generation_info
        self.expected_attention_config = dict(anchor.constants['ATTENTION_CONFIG'], num_layers=self.layers)
        self.swiglu_config = anchor.constants['SWIGLU_CONFIG']
        self.anchor_info = {'reader_sha256': ANCHOR_READER_SHA, 'audit_sha256': sha(anchor_audit),
                            'comparisons': anchor_comparisons, 'performance': anchor_perf,
                            'snapshot_files_available': anchor.snapshot_present,
                            'snapshot_files_missing': anchor.snapshot_missing, 'raw_backup': anchor.backup_info}
        self.fresh_proofs(anchor)
        self.provenance = {'frozen_sources_sha256': self.frozen_hashes, 'controller_sources_sha256': self.controllers,
                           'backend': self.backend, 'probe_path': REMOTE + '/probe-full-' + PROBE_TRIAL + '.json',
                           'probe_sha256': self.probe_info['sha256'], 'probe_trial_label': PROBE_TRIAL,
                           'candidate_version': self.version,
                           'historical_probe_sha256': FROZEN['stage31/probe-full-003.json'],
                           'verifier_path': REMOTE_ATTENTION + '/probe_full.py',
                           'verifier_sha256': self.probe_info['verifier_sha256'],
                           'native_capture_receipt_sha256': sha(self.frozen('stage31/native-default-capture-02/attention.json')),
                           'native_capture_raw_files': anchor.capture['raw_files'],
                           'candidate_path': REMOTE_ATTENTION + '/candidate-003.py', 'candidate_sha256': CANDIDATE_SHA,
                           'swiglu_candidate_sha256': None, 'native_source_commit': COMMIT, 'native_source_file_count': 600,
                           'adaptation': 'Only declared larger model/seed/path/outer launcher identities and two '
                                         'controlled-left candidate identity gates adapted. Original all-tensor, '
                                         'master, source, precision, optimizer and timer gates retained. Default '
                                         'native compiled SwiGLU and fused causal softmax in every arm; no Swi004.'}

    def fresh_proofs(self, anchor):
        for name, digest in FRESH_PROOFS.items():
            require(sha(self.raw(name)) == digest, 'fresh declared proof bytes changed: ' + name)
        backend_path = self.raw(CURRENT_BACKEND)
        backend = read(backend_path)
        expected = anchor.backend['native_source_sha256']
        require(backend['schema'] == 'native-attention-softmax-backend-probe-v1'
                and backend['status'] == 'passed' and backend['requested_backend'] == 'default-fusion'
                and backend['sources_unchanged'] is True and backend['megatron_commit'] == COMMIT
                and backend['source_manifest_sha256'] == sha(FORMAL / 'source_manifest.json')
                and backend['controller_sha256'] == FROZEN['stage31/probe_native_backend.py']
                and backend['native_source_sha256'] == expected
                and len(backend['cases']) == 3 and {row['sequence'] for row in backend['cases']} == {128, 256, 512},
                'fresh default native backend provenance/complete shape evidence differs')
        extensions = {}
        for row in backend['cases']:
            require(row['status'] == 'passed' and row['route'] == 'cuda-extension'
                    and all(row[key] is True for key in
                            ('kernel_available', 'forward_completed', 'backward_completed',
                             'cuda_extension_forward_completed', 'cuda_extension_backward_completed',
                             'scaled_masked_softmax_fusion', 'finite_output', 'finite_gradient', 'mask_unchanged'))
                    and row['softmax_in_fp32'] is False and row['additional_softmax_scale'] is None
                    and row['softmax_offset'] is None and row['input_dtype'] == 'torch.bfloat16',
                    'fresh default fused softmax forward/backward route was not proved')
            for name in ('scaled_masked_softmax_cuda', 'scaled_upper_triang_masked_softmax_cuda'):
                item = row['extensions_after'][name]
                require(item['loaded'] is item['importable'] is True, 'fresh native binary unavailable')
                value = {'path': item['path'], 'sha256': item['sha256']}
                require(extensions.setdefault(name, value) == value, 'fresh native binary differs between shapes')
        require(extensions == anchor.backend['native_extensions'],
                'fresh Pod must retain original source and CoreX binary identities')
        self.backend = {'path': REMOTE + '/' + CURRENT_BACKEND, 'sha256': sha(backend_path),
                        'native_source_sha256': expected, 'native_extensions': extensions,
                        'historical_path': anchor.backend['path'], 'historical_sha256': anchor.backend['sha256']}
        probe_path = self.raw('probe-full-' + PROBE_TRIAL + '.json')
        probe = read(probe_path)
        require(probe['schema'] == 'native-fused-full-attention-independent-probe-v1'
                and probe['status'] == 'passed' and probe['version'] == PROBE_TRIAL
                and probe['candidate_sha256'] == CANDIDATE_SHA
                and probe['verifier_sha256'] == FROZEN['stage31/probe_full.py']
                and probe['native_source_sha256'] == expected
                and probe['backend_proof_sha256'] == self.backend['historical_sha256']
                and probe['atol'] == 3e-4 and probe['rtol'] == 1e-3 and probe['comparison'] == 'CPU float64'
                and probe['runtime']['torch'] == '2.4.1'
                and probe['runtime']['selected_megatron_source'] == REMOTE_NATIVE
                and probe['runtime']['optional_backends_disabled_in_process'] == ['transformer_engine', 'apex'],
                'fresh numeric trial must use unchanged003, frozen verifier and original tolerance/runtime')
        require(len(probe['checks']) == 22 and len(probe['guard_checks']) == 2
                and {tuple(row['shape']) for row in probe['checks']} >=
                    {(128, 2, 8, 64), (256, 2, 8, 64), (512, 2, 8, 64)}
                and {row['case'] for row in probe['guard_checks']} == {'cpu', 'float32'}
                and all(row['passed'] is True for row in probe['guard_checks']),
                'fresh full shape and unsupported CPU/FP32 guard coverage differs')
        for row in probe['checks']:
            require(row['passed'] is True and row['inputs_unchanged'] is True
                    and set(row['checks']) == {'output', 'query_grad', 'key_grad', 'value_grad'},
                    'full output/QKV gradients and immutability required')
            for check in row['checks'].values():
                require(check['passed'] is True and check['bad_elements'] == 0
                        and math.isfinite(check['max_abs']) and check['max_abs'] >= 0,
                        'fresh probe numerical subcheck contradicts pass')
        require(set(probe['native_capture_replay']) == {'output', 'query_grad', 'key_grad', 'value_grad'}
                and all(row['passed'] is True and row['exact_equal'] is True and row['bad_elements'] == 0
                        for row in probe['native_capture_replay'].values()), 'fresh native capture replay must be exact')
        for row in probe['timing']:
            require(row['mode'] in ('forward', 'forward-backward') and set(row['raw_pairs']) == {'native', 'candidate'},
                    'fresh independent Event microtiming scope differs')
            for arm in ('native', 'candidate'):
                values = row['raw_pairs'][arm]
                require(len(values) == 12 and all(math.isfinite(value) and value > 0 for value in values),
                        'invalid fresh Event raw samples')
                close(row[arm + '_median_ms'], statistics.median(values), 'fresh Event median differs from raw samples')
        self.probe_info = {'checks': 22, 'guards': 2, 'trial_label': PROBE_TRIAL, 'candidate_version': self.version,
                           'sha256': sha(probe_path), 'verifier_sha256': probe['verifier_sha256']}

    def backup(self):
        path = self.base / f'raw-inventory-{self.model}-{self.args.audit_suffix}.json'
        receipt_path = self.base / f'raw-backup-{self.model}-{self.args.audit_suffix}-receipt.json'
        require(not receipt_path.exists() or path.exists(), 'backup receipt lacks inventory')
        if not path.exists():
            return
        rows = read(path)
        require(isinstance(rows, list) and rows, 'nonempty raw inventory required')
        folders = {f'{self.model}-{arm}-audit-003-{self.args.audit_suffix}' for arm in ('native', 'control', 'optimized')}
        unique, present, missing = {}, 0, 0
        for row in rows:
            rel = relative(row['path'])
            require(len(rel.parts) >= 3 and rel.parts[0] in folders and rel.parts[1] == 'snapshots'
                    and type(row['bytes']) is int and row['bytes'] >= 0 and row['path'] not in self.inventory
                    and re.fullmatch(r'[0-9a-f]{64}', row['sha256']) is not None,
                    'raw inventory escaped model/round or duplicates a path')
            self.inventory[row['path']] = row
            require(unique.setdefault(row['sha256'], row)['bytes'] == row['bytes'], 'same SHA has conflicting size')
            raw = self.raw_snapshot(row['path'])
            if raw.is_file():
                self.byte_check(raw, row)
                if receipt_path.exists():
                    object_path = self.base / 'raw-backup/objects' / row['sha256']
                    require(object_path.is_file() and raw.samefile(object_path), 'raw path is not a deduplicated hardlink')
                present += 1
            else:
                missing += 1
        require({relative(row['path']).parts[0] for row in rows} == folders,
                'raw inventory must cover all three arms of this model/round')
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt['schema'] == 'frozen-attention-stage32-raw-backup-v1' and receipt['status'] == 'verified'
                    and receipt['model_name'] == self.model and receipt['version'] == '003'
                    and receipt['audit_suffix'] == self.args.audit_suffix and receipt['remote_root'] == REMOTE
                    and receipt['remote_deleted'] is False and receipt['inventory_sha256'] == sha(path)
                    and receipt['files'] == rows and receipt['verified_files'] == len(rows)
                    and receipt['raw_bytes'] == sum(row['bytes'] for row in rows)
                    and receipt['unique_files'] == len(unique) and receipt['unique_bytes'] == sum(row['bytes'] for row in unique.values())
                    and receipt['deduplication'] == 'shared SHA256 objects and hardlinks'
                    and receipt['transfer_archive_retained'] is False
                    and receipt['backup_script_sha256'] == sha(SOURCE / 'backup_raw.py')
                    and receipt['gpu_timing_finished_acknowledged'] is True,
                    'raw backup receipt contradicts immutable inventory')
        self.backup_info = {'inventory_available': True, 'receipt_available': receipt_path.exists(),
                            'inventory_sha256': sha(path), 'inventory_files': len(rows),
                            'files_available': present, 'files_missing': missing, 'unique_contents': len(unique)}

    def expected_command(self, folder, arm, steps, seed, python):
        root = REMOTE + '/' + folder.name
        result = [python, '-u', REMOTE + '/launch_attention_scale.py', '--attention-receipt', root + '/attention.json',
                  '--swiglu-receipt', root + '/swiglu.json', '--delegate-launcher', REMOTE_GRAD + '/launch_gradnorm-04.py',
                  '--bf16-root', REMOTE_BF, '--gradnorm-receipt', root + '/gradnorm.json',
                  '--megatron-root', REMOTE_NATIVE, '--contract', REMOTE + '/contracts/' + self.model + '.json',
                  '--entry-evidence', root + '/entry.json']
        if arm != 'native':
            result += ['--candidate', REMOTE_BF + '/candidates/0008/candidate.py',
                       '--gradnorm-candidate', REMOTE_GRAD + '/candidate.py']
        if arm == 'optimized':
            result += ['--attention-candidate', REMOTE_ATTENTION + '/candidate-003.py']
        if steps == 3:
            result += ['--audit-dir', root + '/snapshots']
        return result + self.contract['training_argv'] + self.training_suffix(folder, steps, seed)

    def training_suffix(self, folder, steps, seed):
        result = ['--train-iters', str(steps), '--seed', str(seed), '--log-interval', '1' if steps == 3 else '10']
        if steps == 3:
            result += ['--save', REMOTE + '/' + folder.name + '/snapshots', '--save-interval', '3',
                       '--save-wgrads-interval', '1', '--save-params-interval', '1']
        return result


def runtime_extra(base):
    """Bind the separately restored original dataset build support, without importing it."""
    before_path = base / 'runtime-extra-before-perf.json'
    after_path = base / 'runtime-extra-after-perf.json'
    before, after = read(before_path), read(after_path)
    expected = {
        'megatron/core/datasets/Makefile': {
            'bytes': 313, 'sha256': '0555c53484b55bf1a7e40aa4d2e09329f6767512282d428e2c868c677ed50350'},
        'megatron/core/datasets/helpers.cpp': {
            'bytes': 29803, 'sha256': 'dbc3429a1dac7a86946494a61d14fe5754538399e6413ca6939de46cc530b24b'},
        'megatron/core/datasets/helpers_cpp.cpython-310-x86_64-linux-gnu.so': {
            'bytes': 248352, 'sha256': '8ac8bb85f0f61c2589c3ea76447fd5c22708077bbcad0c83b73bdcfcb2b6cf38'},
    }
    require(before['phase'] == 'before-performance' and after['phase'] == 'after-performance'
            and before['megatron_commit'] == after['megatron_commit']
                == '5be9626709af2722333bf54797c954c09edeada3'
            and before['files'] == after['files'] == expected,
            'separately restored native dataset source/build binary changed during timing')
    return {'before_sha256': sha(before_path), 'after_sha256': sha(after_path),
            'reported_remote_files': expected, 'reported_bytes_unchanged': True,
            'local_binary_bytes_verified': False,
            'scope': 'Remote before/after SHA reports for three extra native dataset build files; '
                     'supplements the unchanged600-file manifest, never seals all installed binaries.'}


def verify(args):
    archive = Archive(args)
    archive.foundations()
    if args.foundations_only:
        return {'schema': 'native-full-attention-stage32-cpu-admission-verification-v1',
                'status': 'admission_verified', 'model_name': args.model, 'version': args.version,
                'candidate_sha256': archive.candidate_sha, 'contract_sha256': sha(archive.contract_path),
                'controller_sources_sha256': archive.controllers, 'frozen_sources_checked': len(FROZEN),
                'fresh_proofs_sha256': FRESH_PROOFS, 'probe': archive.probe_info,
                'historical_stage31_admission': archive.anchor_info,
                'stage32_training_audit_verified': False, 'stage32_performance_verified': False,
                'gpu_rerun': False, 'tensor_arithmetic_replayed': False,
                'scope': 'Only immutable source/model/runtime/fresh independent probe admission verified. '
                         'Historical valid003 promotion is checked as admission, never as a larger-model result.'}
    archive.backup()
    extra = runtime_extra(archive.base)
    path, comparisons, routes = archive.audit()
    performance = archive.performance(path, routes)
    backup = archive.backup_info
    complete = (performance['available'] is True and backup['receipt_available'] is True
                and backup['files_missing'] == 0 and archive.snapshot_missing == 0
                and set(archive.snapshot_records).issubset(archive.inventory))
    require(not args.require_complete or complete,
            'complete evidence requires validated performance and every inventoried raw audit snapshot/backup receipt')
    return {'schema': 'native-full-attention-stage32-cpu-archive-verification-v1', 'status': 'verified',
            'model_name': args.model, 'version': args.version, 'audit_suffix': args.audit_suffix,
            'run_suffix': args.run_suffix, 'candidate_sha256': archive.candidate_sha,
            'contract_sha256': sha(archive.contract_path), 'controller_sources_sha256': archive.controllers,
            'frozen_sources_checked': len(FROZEN), 'verifier_sha256': sha(__file__),
            'fresh_proofs_sha256': FRESH_PROOFS, 'probe': archive.probe_info,
            'generation': archive.generation_info, 'historical_stage31_admission': archive.anchor_info,
            'audit_sha256': sha(path), 'comparisons': comparisons, 'untimed_audit_routes': routes,
            'performance': performance, 'evidence_complete': complete,
            'snapshot_inventory_records': len(archive.snapshot_records),
            'snapshot_files_available': archive.snapshot_present, 'snapshot_files_missing': archive.snapshot_missing,
            'raw_backup': backup, 'native_extension_binary_bytes_verified_locally': False,
            'runtime_extra': extra,
            'gpu_rerun': False, 'tensor_arithmetic_replayed': False,
            'scope': 'All recorded source/contract/controller/fresh probe/generation identities, three-way full '
                     'model/loss/all-gradient/BF16-parameter/FP32-master receipts and available original '
                     '120/20 native training interval logs rechecked on CPU. Available raw bytes are SHA256 '
                     'checked; missing backups remain explicit. Fresh backend and actual audit/formal receipts '
                     'bind native CoreX extension hashes, without claiming local binary bytes. No Torch or '
                     'candidate import, GPU rerun, tensor arithmetic replay, JIT GPU timeline or production extrapolation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(MODELS), required=True)
    parser.add_argument('--version', choices=('003',), default='003')
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    parser.add_argument('--foundations-only', action='store_true',
                        help='Check immutable admission only; no new training or performance result is accepted.')
    parser.add_argument('--require-complete', action='store_true',
                        help='Require validated performance and every raw audit snapshot/backup receipt.')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    try:
        require(all(re.fullmatch(r'[A-Za-z0-9-]+', value)
                    for value in (args.audit_suffix, args.run_suffix)), 'simple immutable suffixes required')
        require(not (args.foundations_only and args.require_complete), 'admission alone is not complete experiment evidence')
        result = verify(args)
    except Exception as error:
        result = {'status': 'rejected', 'model_name': args.model, 'version': args.version,
                  'gpu_rerun': False, 'tensor_arithmetic_replayed': False,
                  'error_type': type(error).__name__, 'error': str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] in ('verified', 'admission_verified') else 1


if __name__ == '__main__':
    raise SystemExit(main())
