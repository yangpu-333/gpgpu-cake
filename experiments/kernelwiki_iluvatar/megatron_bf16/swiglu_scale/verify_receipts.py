"""Recheck frozen stage30 archives on CPU, without importing Torch or candidates.

Every numerical receipt record and original timing log is checked. Available
raw snapshot bytes are hashed; missing snapshots are reported separately. This
does not deserialize tensors or repeat the original numerical arithmetic.
"""
import argparse
import ast
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


SOURCE = Path(__file__).resolve().parent
BF = SOURCE.parent
FORMAL = BF.parent / 'megatron_training_entry'
SEALED = BF / 'evidence/swiglu-stage29'
REMOTE = '/tmp/kda-swiglu-scale-stage30'
REMOTE_SEALED = '/tmp/kda-swiglu-stage29'
REMOTE_BF = '/tmp/kda-bf16-continuation/megatron_bf16'
REMOTE_FORMAL = '/tmp/kda-bf16-continuation/megatron_training_entry'
REMOTE_NATIVE = '/tmp/kda-bf16-continuation/megatron-native-5be9626'
REMOTE_GRAD = '/tmp/kda-gradnorm-stage26'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
CANDIDATE_SHA = 'a516bb372012cd5569eed99db2815a41c0b229755158289ddf6536248d554250'
PARSER_SHA = '6c2d1f7955f7e474a3de0b9e4edc156bc9ba90b5f23c7a5fce696ee736dcd2d3'
MODELS = {'wide': (4, 256, 4096, 21107200, 30101, [26012, 26013, 26014], 1475, 637),
          'deep': (8, 512, 8192, 42213888, 30201, [26022, 26023, 26024], 2371, 1165)}
ORDERS = [['native', 'control', 'optimized'], ['optimized', 'control', 'native'],
          ['control', 'native', 'optimized']]
EXPECTED_CONFIG = {'gated_linear_unit': True, 'bias_activation_fusion': True,
                   'activation_func': 'torch.nn.functional.silu', 'add_bias_linear': False,
                   'use_te_activation_func': False, 'activation_func_fp8_input_store': False,
                   'glu_linear_offset': 0.0, 'activation_func_clamp_value': None,
                   'cpu_offloading': False, 'per_token_scale_present': False}


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
            'unsafe archive relative path: ' + str(value))
    return path


class Archive:
    def __init__(self, args):
        self.args = args
        self.base = args.base.resolve()
        self.model = args.model
        (self.layers, self.sequence, self.vocabulary, self.parameters, self.audit_seed,
         self.seeds, self.check_count, self.tensor_count) = MODELS[self.model]
        self.contract_path = self.raw('contracts/' + self.model + '.json')
        self.contract = read(self.contract_path)
        self.adaptation_path = self.raw('adaptation-record.json')
        self.adaptation = read(self.adaptation_path)
        if sha(self.adaptation_path) == sha(SOURCE / 'adaptation-record.json'):
            self.controller_source = SOURCE
        elif (SOURCE / 'controller_v1/adaptation-record.json').is_file() and sha(self.adaptation_path) == sha(SOURCE / 'controller_v1/adaptation-record.json'):
            self.controller_source = SOURCE / 'controller_v1'
        else:
            raise ValueError('archive adaptation has no exact current or frozen-v1 controller identity')
        self.manifest = read(FORMAL / 'source_manifest.json')
        self.inventory = {}
        self.snapshot_records = {}
        self.hashed = {}
        self.snapshot_present = self.snapshot_missing = 0
        self.backup_info = {'inventory_available': False, 'receipt_available': False,
                            'inventory_files': 0, 'files_available': 0, 'files_missing': 0}

    def raw(self, name):
        rel = relative(name)
        result = self.base.joinpath(*rel.parts)
        require(result.resolve().is_relative_to(self.base), 'archive path escaped root')
        require(result.is_file(), 'missing small evidence: ' + str(result))
        return result

    def recorded(self, value):
        value = str(value).replace('\\', '/')
        roots = ((REMOTE, self.base), (REMOTE_SEALED, SEALED), (REMOTE_BF, BF),
                 (REMOTE_FORMAL, FORMAL))
        for remote, local in roots:
            if value.startswith(remote + '/'):
                rel = relative(value[len(remote) + 1:])
                result = local.joinpath(*rel.parts)
                require(result.resolve().is_relative_to(local.resolve()), 'recorded path escaped root')
                return result
        if value == REMOTE_GRAD + '/candidate.py':
            return BF / 'gradnorm_candidates/001/candidate.py'
        if value == REMOTE_GRAD + '/launch_gradnorm-04.py':
            return BF / 'evidence/gradnorm-stage26/launch_gradnorm-04.py'
        raise ValueError('unmapped recorded source path: ' + value)

    def frozen(self, name):
        group, rest = name.split('/', 1)
        if group == 'gradnorm':
            return self.recorded(REMOTE_GRAD + '/' + rest)
        roots = {'stage29': SEALED, 'bf16': BF, 'formal': FORMAL}
        require(group in roots, 'unmapped frozen dependency: ' + name)
        rel = relative(rest)
        return roots[group].joinpath(*rel.parts)

    def byte_check(self, path, row):
        stat = path.stat()
        require(stat.st_size == row['bytes'], 'raw byte size differs: ' + str(path))
        key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if key not in self.hashed:
            self.hashed[key] = sha(path)
        require(self.hashed[key] == row['sha256'], 'raw bytes differ: ' + str(path))

    def raw_snapshot(self, path):
        rel = relative(path)
        choices = [self.base.joinpath(*rel.parts),
                   (self.base / 'raw-backup').joinpath(*rel.parts)]
        return next((item for item in choices if item.is_file()), choices[0])

    def foundations(self):
        record = self.adaptation
        require(record['schema'] == 'fixed-swiglu-stage30-adaptation-v1'
                and record['candidate'] == '004' and record['candidate_sha256'] == CANDIDATE_SHA,
                'fixed-code adaptation identity differs')
        expected_controllers = {'common.py', 'run_audit.py', 'run_performance.py',
                                'contracts/wide.json', 'contracts/deep.json'}
        require(set(record['controller_sources_sha256']) == expected_controllers,
                'complete fixed stage30 controller/contract set required')
        for name, digest in record['controller_sources_sha256'].items():
            archived = self.raw(name)
            require(sha(archived) == digest and sha(self.controller_source / name) == digest,
                    'archived/project controller bytes differ: ' + name)
        require(sha(self.controller_source / 'adaptation-record.json') == sha(self.adaptation_path),
                'project/deployed adaptation ledger differs')
        for name, digest in record['frozen_sources_sha256'].items():
            require(sha(self.frozen(name)) == digest, 'frozen dependency differs: ' + name)
        require(len(record['frozen_sources_sha256']) == 18,
                'complete 18-dependency frozen ledger required')
        require(self.manifest['commit'] == self.contract['megatron_commit']
                == record['native_source_commit'] == COMMIT
                and len(self.manifest['files']) == record['native_source_file_count'] == 600,
                'complete fixed 600-file native source manifest required')
        original_path = SEALED / ('contracts/' + self.model + '.json')
        original = read(original_path)
        for key in ('training_argv', 'model', 'candidate_sha256', 'correctness'):
            require(self.contract[key] == original[key], 'original model contract changed: ' + key)
        stage = self.contract['stage30']
        require(stage['model_name'] == self.model and stage['source_contract_sha256'] == sha(original_path)
                and stage['swiglu_candidate_sha256'] == CANDIDATE_SHA
                and stage['audit_seed'] == self.audit_seed
                and stage['rmsnorm_candidate_enabled'] is False
                and stage['heldout_code_retuned'] is False, 'heldout scope provenance differs')
        cfg = self.contract['model']
        expected = {'num_layers': self.layers, 'sequence_length': self.sequence,
                    'vocab_size': self.vocabulary, 'parameters': self.parameters,
                    'hidden_size': 512, 'ffn_hidden_size': 2048, 'num_attention_heads': 8,
                    'micro_batch_size': 2, 'global_batch_size': 2, 'bf16': True,
                    'normalization': 'RMSNorm', 'epsilon': 1e-5,
                    'fp32_residual_connection': False, 'gated_linear_unit': True,
                    'bias_activation_fusion': True}
        require(all(cfg.get(key) == value for key, value in expected.items()),
                'fixed heldout model/precision/default fusion differs')
        require(self.contract['swiglu']['primary_shape'] == [self.sequence, 2, 4096]
                and self.contract['correctness']['steps'] == 3
                and self.contract['correctness']['atol'] == 3e-4
                and self.contract['correctness']['rtol'] == 1e-3,
                'fixed actual activation shape/numerical contract differs')
        perf = self.contract['performance']
        require(perf['steps'] == 120 and perf['independent_seeds'] == self.seeds
                and perf['arms'] == ['native', 'control', 'optimized']
                and perf['process_orders'] == ORDERS and perf['tensor_snapshotting'] is False
                and '--swiglu' in self.contract['training_argv']
                and '--no-bias-swiglu-fusion' not in self.contract['training_argv'],
                'fixed native timer/rotation/default SwiGLU contract differs')
        require(sha(SEALED / 'candidate-004.py') == CANDIDATE_SHA
                and sha(BF / 'candidates/0008/candidate.py') == self.contract['candidate_sha256'],
                'fixed SwiGLU004/CE0008 candidate differs')
        main, probe = read(SEALED / 'decision.json'), read(SEALED / 'probe-004-v3.json')
        require(main['promoted'] is True and main['selected_candidate'] == '004'
                and probe['status'] == 'passed' and probe['candidate_sha256'] == CANDIDATE_SHA
                and len(probe['checks']) == 37 and len(probe['guard_checks']) == 4
                and all(check['passed'] is True for row in probe['checks'] for check in row['checks'])
                and all(row['passed'] is True for row in probe['guard_checks']),
                'frozen main-model candidate admission evidence differs')

    def backup(self):
        path = self.base / ('raw-inventory-' + self.model + '.json')
        receipt_path = self.base / ('raw-backup-' + self.model + '-receipt.json')
        require(not receipt_path.exists() or path.exists(), 'backup receipt lacks inventory')
        if not path.is_file():
            return
        rows = read(path)
        require(isinstance(rows, list) and bool(rows), 'nonempty raw inventory required')
        unique = {}
        present = missing = 0
        for row in rows:
            rel = relative(row['path'])
            require(rel.parts[0].startswith(self.model + '-') and 'snapshots' in rel.parts
                    and row['bytes'] >= 0 and isinstance(row['sha256'], str)
                    and len(row['sha256']) == 64 and row['path'] not in self.inventory,
                    'raw inventory scope or duplicate path differs')
            self.inventory[row['path']] = row
            previous = unique.setdefault(row['sha256'], row)
            require(previous['bytes'] == row['bytes'], 'same SHA has conflicting byte count')
            raw = self.raw_snapshot(row['path'])
            if raw.is_file():
                self.byte_check(raw, row)
                present += 1
            else:
                missing += 1
        receipt_checked = False
        if receipt_path.is_file():
            receipt = read(receipt_path)
            require(receipt['status'] == 'verified' and receipt['model_name'] == self.model
                    and receipt['remote_root'] == REMOTE and receipt['remote_deleted'] is False
                    and receipt['inventory_sha256'] == sha(path) and receipt['files'] == rows
                    and receipt['verified_files'] == len(rows)
                    and receipt['raw_bytes'] == sum(row['bytes'] for row in rows)
                    and receipt['unique_files'] == len(unique)
                    and receipt['unique_bytes'] == sum(row['bytes'] for row in unique.values())
                    and receipt['deduplication'] == 'shared SHA256 objects and hardlinks'
                    and receipt['transfer_archive_retained'] is False
                    and receipt['backup_script_sha256'] == sha(SOURCE / 'backup_raw.py')
                    and receipt['gpu_timing_finished_acknowledged'] is True,
                    'raw backup receipt contradicts inventory')
            receipt_checked = True
        self.backup_info = {'inventory_available': True, 'inventory_sha256': sha(path),
                            'receipt_available': receipt_path.is_file(),
                            'receipt_checked': receipt_checked, 'inventory_files': len(rows),
                            'files_available': present, 'files_missing': missing,
                            'unique_contents': len(unique)}

    def expected_command(self, folder, arm, steps, seed, python):
        remote_folder = REMOTE + '/' + folder.name
        result = [python, '-u', REMOTE_SEALED + '/launch_swiglu-03.py',
                  '--swiglu-receipt', remote_folder + '/swiglu.json',
                  '--gradnorm-launcher', REMOTE_GRAD + '/launch_gradnorm-04.py',
                  '--bf16-root', REMOTE_BF, '--gradnorm-receipt', remote_folder + '/gradnorm.json',
                  '--megatron-root', REMOTE_NATIVE, '--contract',
                  REMOTE + '/contracts/' + self.model + '.json',
                  '--entry-evidence', remote_folder + '/entry.json']
        if arm != 'native':
            result += ['--candidate', REMOTE_BF + '/candidates/0008/candidate.py',
                       '--gradnorm-candidate', REMOTE_GRAD + '/candidate.py']
        if arm == 'optimized':
            result += ['--swiglu-candidate', REMOTE_SEALED + '/candidate-004.py']
        if steps == 3:
            result += ['--audit-dir', remote_folder + '/snapshots']
        result += self.contract['training_argv'] + ['--train-iters', str(steps), '--seed', str(seed),
                                                    '--log-interval', '1' if steps == 3 else '10']
        if steps == 3:
            result += ['--save', remote_folder + '/snapshots', '--save-interval', '3',
                       '--save-wgrads-interval', '1', '--save-params-interval', '1']
        return result

    def entry(self, folder, arm, steps, seed):
        command, entry = read(folder / 'command.json'), read(folder / 'entry.json')
        require(command['case'] == folder.name and command['stage30_arm'] == arm
                and command['model_name'] == self.model and command['steps'] == steps
                and command['seed'] == seed and command['exit_code'] == 0
                and command['audit'] is (steps == 3) and command['profile'] is False
                and command['mode'] == ('native' if arm == 'native' else 'optimized')
                and command['contract_sha256'] == entry['contract_sha256'] == sha(self.contract_path)
                and command['common_sha256'] == sha(self.raw('common.py'))
                and command['wrapper_sha256'] == sha(SEALED / 'launch_swiglu-03.py')
                and command['swiglu_candidate_sha256'] == (CANDIDATE_SHA if arm == 'optimized' else None)
                and command['adaptation_record_sha256'] == sha(self.adaptation_path)
                and entry['status'] == 'completed', 'successful fixed-contract process headers differ')
        require(command['command'][0] == '/root/miniconda3/bin/python'
                and command['command'] == self.expected_command(folder, arm, steps, seed,
                                                                command['command'][0]),
                'exact formal native outer argv differs')
        integration = {name: sha(BF / name) for name in ('launch_training.py', 'kernel_adapter.py', 'run_case.py')}
        integration.update({'formal/' + name: sha(FORMAL / name)
                            for name in ('runtime_compat.py', 'source_manifest.json')})
        require(entry['native_sources_sha256'] == self.manifest['files']
                and entry['megatron_commit'] == COMMIT and entry['integration_sha256'] == integration
                and entry['launcher_sha256'] == integration['launch_training.py']
                and entry['formal_entry'] == REMOTE_NATIVE + '/pretrain_gpt.py'
                and entry['profile_hooks'] == [] and 'profiler_activities' not in entry,
                'all 600 native sources/frozen integration/original entry differs')
        require(entry['runtime_compatibility']['torch'] == '2.4.1'
                and entry['runtime_compatibility']['optional_backends_disabled_in_process']
                == ['transformer_engine', 'apex'], 'shared runtime compatibility differs')
        require(len(entry['models']) == 1, 'exactly one native GPT model chunk required')
        expected = {'parameters': self.parameters, 'layers': self.layers, 'hidden_size': 512,
                    'vocab_size': self.vocabulary, 'bf16': True, 'fp16': False,
                    'class': 'megatron.core.models.gpt.gpt_model.GPTModel'}
        require(all(entry['models'][0].get(key) == value for key, value in expected.items()),
                'actual constructed native model differs')
        require(entry['iluvatar_kernels'] is (arm != 'native'), 'CE opt-in scope differs')
        if arm == 'native':
            require(entry['kernel_adapters'] == [], 'native installed a CE adapter')
        else:
            require(len(entry['kernel_adapters']) == 1, 'exactly one fixed CE adapter required')
            adapter = entry['kernel_adapters'][0]
            require(all(adapter[key] is True for key in ('enabled', 'active', 'provenance_verified'))
                    and adapter['candidate_sha256'] == adapter['candidate_expected_sha256']
                    == self.contract['candidate_sha256'] and adapter['runtime_world_size'] == 1
                    and adapter['verified_devices'] in ({'0': 'BI-V150'}, {'0': 'Iluvatar BI-V150'})
                    and adapter['counts'] == {'ce_candidate_calls': steps, 'ce_native_calls': 0,
                                              'residual_native_calls': self.layers * steps,
                                              'fallback_reasons': {}},
                    'actual CE source/device/coverage differs')
        suffix = ['--train-iters', str(steps), '--seed', str(seed), '--log-interval', '1' if steps == 3 else '10']
        if steps == 3:
            suffix += ['--save', REMOTE + '/' + folder.name + '/snapshots', '--save-interval', '3',
                       '--save-wgrads-interval', '1', '--save-params-interval', '1']
        require(entry['training_argv'] == self.contract['training_argv'] + suffix,
                'actual native training argv differs')
        require(datetime.fromisoformat(command['started_utc']) < datetime.fromisoformat(command['finished_utc'])
                and math.isfinite(command['whole_process_elapsed_seconds'])
                and command['whole_process_elapsed_seconds'] > 0, 'process chronology differs')
        return command, entry

    def hooks(self, folder, arm, steps):
        activation, grad = read(folder / 'swiglu.json'), read(folder / 'gradnorm.json')
        enabled, use_grad = arm == 'optimized', arm != 'native'
        calls = self.layers * steps
        native_sources = {name: self.manifest['files'][name] for name in
                          ('megatron/core/transformer/mlp.py',
                           'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
        require(activation['enabled'] is enabled and activation['calls'] == (calls if enabled else 0)
                and activation['native_calls'] == activation['fallback_calls'] == 0
                and activation['wrapper_sha256'] == sha(SEALED / 'launch_swiglu-03.py')
                and activation['candidate_sha256'] == (CANDIDATE_SHA if enabled else None)
                and activation['native_source_sha256'] == native_sources
                and activation['native_compiled_callable'] is True
                and activation.get('extension') is None,
                'fixed activation source/default fused native/dispatch scope differs')
        key = str(((self.sequence, 2, 4096), (8192, 4096, 1), 'torch.bfloat16'))
        require(activation['shapes'] == ({key: calls} if enabled else {}),
                'actual activation shape/layout/count differs')
        if enabled or steps == 3:
            require(activation['configuration'] == EXPECTED_CONFIG, 'native activation configuration differs')
        require(not activation.get('capture_written', False), 'unexpected extra activation tensor capture')
        require(grad['enabled'] is use_grad and grad['calls'] == (steps if use_grad else 0)
                and grad['fallback_calls'] == 0
                and grad['wrapper_sha256'] == sha(BF / 'evidence/gradnorm-stage26/launch_gradnorm-04.py')
                and grad['candidate_sha256'] == (sha(BF / 'gradnorm_candidates/001/candidate.py') if use_grad else None)
                and len(grad['norm_audit']) == (3 if use_grad and steps == 3 else 0),
                'fixed gradnorm source/dispatch/audit scope differs')
        for row in grad['norm_audit']:
            require(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm'],
                    'actual training gradient norm differs')
        if steps == 120:
            require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation')),
                    'diagnostic artifacts in timed case')
        return {'swiglu': activation, 'gradnorm': grad,
                'raw_sha256': {name: sha(folder / name) for name in ('swiglu.json', 'gradnorm.json')}}

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
        for row in proof['input_files']:
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
        exact = all(row['exact_equal'] is True for row in tensors)
        bitwise = all(row['native_sha256'] == row['optimized_sha256'] for row in tensors)
        return {'checks': counts['total'], 'tensors': len(tensors), 'tensor_groups': groups,
                'raw_value_comparisons': len(proof['raw_value_comparisons']),
                'all_tensors_exact_equal': exact, 'all_tensor_bytes_equal': bitwise}

    def audit(self):
        suffix = self.args.audit_suffix
        path = self.raw('audit-summary-' + self.model + '-004-' + suffix + '.json')
        audit = read(path)
        require(audit['schema'] == 'fixed-swiglu-stage30-three-way-audit-v1'
                and audit['status'] == 'completed' and audit['passed'] is True
                and audit['model_name'] == self.model and audit['suffix'] == suffix
                and audit['audit_seed'] == self.audit_seed and audit['candidate_sha256'] == CANDIDATE_SHA
                and audit['contract_sha256'] == sha(self.contract_path)
                and audit['wrapper_sha256'] == sha(SEALED / 'launch_swiglu-03.py')
                and audit['common_sha256'] == sha(self.raw('common.py'))
                and audit['runner_sha256'] == sha(self.raw('run_audit.py'))
                and audit['adaptation_record_sha256'] == sha(self.adaptation_path)
                and audit['frozen_sources_sha256'] == self.adaptation['frozen_sources_sha256']
                and audit['frozen_comparator_sha256'] == sha(BF / 'compare_training.py'),
                'same-model numerical audit provenance differs')
        require(set(audit['comparisons']) == {'controlled', 'native_optimized', 'native_control'}
                and [row['arm'] for row in audit['completed_cases']] == ['native', 'control', 'optimized'],
                'complete ordered three-way audit required')
        folders, commands = {}, {}
        for item in audit['completed_cases']:
            arm = item['arm']
            folder = self.recorded(item['path'])
            require(folder.parent == self.base
                    and folder.name == self.model + '-' + arm + '-audit-004-' + suffix,
                    'unexpected numerical case root/name')
            command, entry = self.entry(folder, arm, 3, self.audit_seed)
            require(item['hooks'] == self.hooks(folder, arm, 3), 'audit summary hook receipts differ')
            folders[arm], commands[arm] = folder, command
        for first, second in zip(('native', 'control'), ('control', 'optimized')):
            require(datetime.fromisoformat(commands[first]['finished_utc'])
                    <= datetime.fromisoformat(commands[second]['started_utc']), 'audits overlap or order differs')
        labels = {'controlled': ('control', 'optimized'), 'native_optimized': ('native', 'optimized'),
                  'native_control': ('native', 'control')}
        results = {}
        for label, (left, right) in labels.items():
            item = audit['comparisons'][label]
            report_path = self.recorded(item['path'])
            require(report_path.parent == self.base and sha(report_path) == item['sha256'],
                    'comparison summary points to different raw bytes')
            proof = read(report_path)
            require(proof['stage30_comparison'] == label and proof['model_name'] == self.model
                    and proof['native_case'] == REMOTE + '/' + folders[left].name
                    and proof['optimized_case'] == REMOTE + '/' + folders[right].name
                    and proof['steps'] == 3 and proof['atol'] == 3e-4 and proof['rtol'] == 1e-3
                    and proof['comparison_device'] == 'CPU float64 copies'
                    and proof['comparator_sha256'] == audit['frozen_comparator_sha256']
                    and proof['stage30_runner_sha256'] == audit['runner_sha256']
                    and proof['adaptation_record_sha256'] == sha(self.adaptation_path)
                    and proof['swiglu_candidate_sha256'] == CANDIDATE_SHA,
                    'raw numerical comparison scope/provenance differs')
            results[label] = self.tensors(proof)
            require(item['passed'] is True and item['check_summary'] == proof['check_summary']
                    and item['tensor_comparisons'] == results[label]['tensors']
                    and item['all_tensors_exact_equal'] is results[label]['all_tensors_exact_equal']
                    and item['all_tensor_bytes_equal'] is results[label]['all_tensor_bytes_equal'],
                    'comparison summary contradicts complete raw records')
            require(results[label]['all_tensors_exact_equal'] and results[label]['all_tensor_bytes_equal'],
                    'published stage30 bitwise outcome not supported by every raw tensor')
        for rel, row in self.snapshot_records.items():
            if self.raw_snapshot(rel).is_file():
                self.snapshot_present += 1
            else:
                self.snapshot_missing += 1
        return path, audit, results

    def load_validator(self):
        parser_path = FORMAL / 'summarize_performance.py'
        validator_path = BF / 'summarize_performance.py'
        require(sha(parser_path) == PARSER_SHA, 'original standard-library native interval parser differs')
        # These frozen modules contain standard-library imports only; reject an
        # unexpected dependency before executing their CPU-only definitions.
        for path in (parser_path, validator_path):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                         else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
                require(all(name.split('.')[0] not in ('torch', 'triton', 'megatron', 'candidate')
                            for name in names), 'GPU/candidate dependency in CPU-only frozen reader')
        spec = importlib.util.spec_from_file_location('stage30_native_interval_parser', parser_path)
        parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser)
        source = validator_path.read_text(encoding='utf-8')
        start = source.index("    invocation=command['command']")
        end = source.index("    runtime=entry['runtime_compatibility']", start)
        replacement = """    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact fixed-code SwiGLU holdout outer wrapper invocation')
    require(entry['formal_entry']==EXPECTED_FORMAL_ENTRY,'original pinned native pretrain entry')
"""
        adapted = source[:start] + replacement + source[end:]
        hashes = {'frozen_validator_sha256': sha(validator_path),
                  'adapted_validator_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
                  'replaced_original_argv_block_sha256': hashlib.sha256(source[start:end].encode()).hexdigest(),
                  'replacement_argv_block_sha256': hashlib.sha256(replacement.encode()).hexdigest()}
        spec = importlib.util.spec_from_file_location('stage30_frozen_validator', validator_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        exec(compile(adapted, str(validator_path), 'exec'), module.__dict__)
        module.EXPECTED_COMMANDS = {}
        module.EXPECTED_FORMAL_ENTRY = REMOTE_NATIVE + '/pretrain_gpt.py'
        return parser, module, hashes

    def performance(self, audit_path):
        path = self.base / ('performance-' + self.model + '-004-' + self.args.run_suffix + '.json')
        if not path.is_file():
            return {'available': False, 'promoted': None, 'timed_processes': 0}
        perf = read(path)
        require(perf['schema'] == 'fixed-swiglu-stage30-three-arm-performance-v1'
                and perf['status'] == 'validated' and perf['model_name'] == self.model
                and perf['run_suffix'] == self.args.run_suffix and perf['candidate_sha256'] == CANDIDATE_SHA
                and perf['contract_sha256'] == sha(self.contract_path)
                and perf['audit_sha256'] == sha(audit_path) and self.recorded(perf['audit_path']) == audit_path
                and perf['wrapper_sha256'] == sha(SEALED / 'launch_swiglu-03.py')
                and perf['runner_sha256'] == sha(self.raw('run_performance.py'))
                and perf['common_sha256'] == sha(self.raw('common.py'))
                and perf['adaptation_record_sha256'] == sha(self.adaptation_path)
                and perf['frozen_sources_sha256'] == self.adaptation['frozen_sources_sha256']
                and perf['ce_candidate_sha256'] == self.contract['candidate_sha256']
                and perf['gradnorm_candidate_sha256'] == sha(BF / 'gradnorm_candidates/001/candidate.py')
                and perf['model'] == self.contract['model'], 'performance source/contract/audit headers differ')
        parser, validator, hashes = self.load_validator()
        require(all(perf['validator_adaptation'][key] == value for key, value in hashes.items()),
                'frozen in-memory validator adaptation differs')
        require(perf['seeds'] == self.seeds and perf['orders'] == ORDERS
                and len(perf['pairs']) == 3 and len(perf['completed_cases']) == 9,
                'complete declared three-seed/nine-process rotation required')
        # All unique numerical input rows (including dependencies and metadata)
        # were rehashed remotely before timing; bind that count to raw receipts.
        audit = read(audit_path)
        inputs = {}
        for item in audit['comparisons'].values():
            for row in read(self.recorded(item['path']))['input_files']:
                previous = inputs.setdefault(row['path'], row)
                require(previous['sha256'] == row['sha256'] and previous['bytes'] == row['bytes'],
                        'same pre-timing input has conflicting numerical receipts')
        require(perf['numerical_input_files_hashed_before_timing'] == len(inputs),
                'pre-timing full raw input rehash coverage differs')
        ratios = {key: [] for key in ('vs_native', 'vs_control', 'control_vs_native')}
        completed = []
        previous_finish = None
        for pair, seed, order in zip(perf['pairs'], self.seeds, ORDERS):
            require(pair['seed'] == seed and pair['order'] == order
                    and set(pair['arms']) == {'native', 'control', 'optimized'}, 'per-seed rotation differs')
            means, models = {}, {}
            for arm in order:
                name = self.model + '-' + arm + '-perf-004-' + self.args.run_suffix + '-' + str(seed)
                folder = self.base / name
                command, entry = self.entry(folder, arm, 120, seed)
                hooks = self.hooks(folder, arm, 120)
                validator.EXPECTED_COMMANDS[str(folder)] = command['command']
                fresh = validator.validate_case(folder, command['mode'], seed, self.contract,
                                                sha(self.contract_path), self.contract['candidate_sha256'])
                fresh.update(stage30_arm=arm, hooks=hooks)
                recorded = pair['arms'][arm]
                require(fresh == recorded, 'published timing/model/header/raw hashes differ from frozen revalidation')
                intervals = parser.parse_intervals((folder / 'stdout.log').read_text(encoding='utf-8'))
                require(intervals == recorded['intervals']
                        and [row['iteration'] for row in intervals if row['iteration'] != 1]
                        == list(range(10, 121, 10))
                        and [row['iteration'] for row in intervals if row['retained']]
                        == list(range(30, 121, 10)), 'original 120-step/first20 exclusion intervals differ')
                means[arm] = statistics.mean(row['mean_iteration_ms'] for row in intervals if row['retained'])
                close(recorded['mean_iteration_ms'], means[arm], 'raw interval mean differs')
                started = datetime.fromisoformat(command['started_utc'])
                finished = datetime.fromisoformat(command['finished_utc'])
                require(previous_finish is None or previous_finish <= started,
                        'timed processes overlap or declared serial rotation differs')
                previous_finish = finished
                models[arm] = entry['models']
                completed.append({'arm': arm, 'seed': seed, 'case': name,
                                  'mean_iteration_ms': means[arm]})
            require(models['native'] == models['control'] == models['optimized'], 'timed models differ')
            values = {'vs_native': means['native'] / means['optimized'],
                      'vs_control': means['control'] / means['optimized'],
                      'control_vs_native': means['native'] / means['control']}
            for metric, value in values.items():
                close(pair[metric], value, 'per-seed speed ratio differs: ' + metric)
                ratios[metric].append(value)
        require(perf['completed_cases'] == completed, 'completed-case summary differs from nine raw cases')
        results = {}
        for metric, values in ratios.items():
            ratio = math.exp(statistics.mean(math.log(value) for value in values))
            results[metric] = {'ratio': ratio, 'gain_percent': 100 * (ratio - 1),
                               'all_seeds_faster': all(value > 1 for value in values)}
            close(perf[metric]['ratio'], ratio, 'geometric mean differs: ' + metric)
            close(perf[metric]['gain_percent'], results[metric]['gain_percent'], 'reported gain differs: ' + metric)
            require(perf[metric]['all_seeds_faster'] is results[metric]['all_seeds_faster'],
                    'all-seed positive flag differs: ' + metric)
        promoted = results['vs_control']['ratio'] > 1.01 and results['vs_control']['all_seeds_faster']
        require(perf['promoted'] is promoted
                and perf['decision'] == ('promoted_tested_model_only' if promoted else 'not_promoted')
                and perf['promotion_rule'] == {'gain_vs_control_strictly_greater_than': 0.01,
                                               'all_three_seeds_faster_vs_control': True},
                'promotion contradicts strict >1% and all-three-seed same-round control gate')
        return {'available': True, 'path': str(path), 'sha256': sha(path),
                'timed_processes': 9, 'run_steps': 1080, 'retained_steps': 900,
                'promoted': promoted, **results}


def verify(args):
    archive = Archive(args)
    archive.foundations()
    archive.backup()
    audit_path, audit, comparisons = archive.audit()
    performance = archive.performance(audit_path)
    return {'schema': 'fixed-swiglu-stage30-cpu-archive-verification-v1', 'status': 'verified',
            'model_name': args.model, 'candidate': '004', 'candidate_sha256': CANDIDATE_SHA,
            'contract_sha256': sha(archive.contract_path), 'audit_sha256': sha(audit_path),
            'controller_source_identity': 'frozen-v1' if archive.controller_source.name == 'controller_v1' else 'current-repair-v2',
            'adaptation_record_sha256': sha(archive.adaptation_path),
            'native_interval_parser_sha256': PARSER_SHA,
            'verifier_sha256': sha(__file__), 'comparisons': comparisons, 'performance': performance,
            'snapshot_inventory_records': len(archive.snapshot_records),
            'snapshot_files_available': archive.snapshot_present,
            'snapshot_files_missing': archive.snapshot_missing, 'raw_backup': archive.backup_info,
            'gpu_rerun': False, 'tensor_arithmetic_replayed': False,
            'scope': 'Every frozen numerical receipt record, source/contract/controller/frozen dependency, '
                     'actual model/shape/config/dispatch header and native interval log is rechecked on CPU. '
                     'Only available raw snapshot bytes are SHA256 checked; missing raw bytes are explicit. '
                     'No Torch/candidate import, tensor arithmetic replay, GPU run or production extrapolation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('wide', 'deep'), required=True)
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    try:
        require(all(value.replace('-', '').isalnum() for value in (args.audit_suffix, args.run_suffix)),
                'simple immutable evidence suffix required')
        result = verify(args)
    except Exception as error:
        result = {'status': 'rejected', 'model_name': args.model, 'gpu_rerun': False,
                  'error_type': type(error).__name__, 'error': str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
