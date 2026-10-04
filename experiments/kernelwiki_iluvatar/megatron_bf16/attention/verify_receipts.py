"""CPU-only stage31 receipt verification, including raw-log performance replay.

No Torch, Triton, candidate or experiment controller is imported. Available
snapshot bytes are hashed; receipt arithmetic is checked, not tensor arithmetic
replayed. Missing raw tensors and native binary bytes are explicitly reported.
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
REMOTE = '/tmp/kda-attention-stage31'
REMOTE_SEALED = '/tmp/kda-swiglu-stage29'
REMOTE_BF = '/tmp/kda-bf16-continuation/megatron_bf16'
REMOTE_FORMAL = '/tmp/kda-bf16-continuation/megatron_training_entry'
REMOTE_NATIVE = '/tmp/kda-bf16-continuation/megatron-native-5be9626'
REMOTE_GRAD = '/tmp/kda-gradnorm-stage26'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
SWIGLU_SHA = 'a516bb372012cd5569eed99db2815a41c0b229755158289ddf6536248d554250'
PARSER_SHA = '6c2d1f7955f7e474a3de0b9e4edc156bc9ba90b5f23c7a5fce696ee736dcd2d3'
SEEDS = [31102, 31103, 31104]
ORDERS = [['native', 'control', 'optimized'], ['optimized', 'control', 'native'],
          ['control', 'native', 'optimized']]


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
    """Read data assignments through AST only; never execute the controller."""
    result = {}
    for node in ast.parse(Path(path).read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Dict):
            keys, values = [], []
            for key, item in zip(value.keys, value.values):
                keys.append(ast.literal_eval(key))
                values.append(result[item.id] if isinstance(item, ast.Name) else ast.literal_eval(item))
            result[node.targets[0].id] = dict(zip(keys, values))
        else:
            try:
                result[node.targets[0].id] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    return result


class Archive:
    def __init__(self, args):
        self.args, self.base, self.version = args, args.base.resolve(), args.version
        self.layers, self.sequence, self.vocabulary, self.parameters = 4, 128, 1024, 17895936
        self.audit_seed, self.seeds, self.check_count, self.tensor_count = 31101, SEEDS, 1475, 637
        self.contract_path = self.raw('contracts/main.json')
        self.contract = read(self.contract_path)
        self.manifest = read(FORMAL / 'source_manifest.json')
        self.constants = literal_constants(self.raw('common.py'))
        self.frozen_hashes = self.constants['FROZEN']
        self.controllers = {name: sha(self.raw(name)) for name in
                            ('common.py', 'launch_attention.py', 'run_audit.py', 'run_performance.py')}
        self.candidate_path = self.raw('candidate-' + self.version + '.py')
        self.candidate_sha = sha(self.candidate_path)
        self.inventory, self.snapshot_records, self.hashed = {}, {}, {}
        self.snapshot_present = self.snapshot_missing = 0
        self.backup_info = {'inventory_available': False, 'receipt_available': False,
                            'inventory_files': 0, 'files_available': 0, 'files_missing': 0}

    def raw(self, name):
        result = self.base.joinpath(*relative(name).parts)
        require(result.resolve().is_relative_to(self.base) and result.is_file(),
                'missing or escaped small evidence: ' + str(result))
        return result

    def recorded(self, value):
        value = str(value).replace('\\', '/')
        for remote, local in ((REMOTE, self.base), (REMOTE_SEALED, SEALED),
                              (REMOTE_BF, BF), (REMOTE_FORMAL, FORMAL)):
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
        roots = {'stage31': REMOTE, 'stage29': REMOTE_SEALED, 'bf16': REMOTE_BF,
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
        require(len(self.frozen_hashes) == 17, 'complete17-source admission ledger required')
        for name, digest in self.frozen_hashes.items():
            require(sha(self.frozen(name)) == digest, 'frozen source changed: ' + name)
        repair_path = SOURCE / 'controller-repair.json'
        repair = read(repair_path)
        old_source = SOURCE / repair['old_source_archive']
        require(old_source.resolve().is_relative_to(SOURCE.resolve())
                and repair['schema'] == 'native-full-attention-stage31-controller-metadata-repair-v1'
                and repair['changed_file'] == 'run_performance.py'
                and sha(old_source) == repair['old_sha256']
                and sha(SOURCE / 'run_performance.py') == repair['new_sha256']
                and old_source.read_text(encoding='utf-8').count(repair['exact_old_block']) == 1
                and old_source.read_text(encoding='utf-8').replace(repair['exact_old_block'], repair['exact_new_block'])
                == (SOURCE / 'run_performance.py').read_text(encoding='utf-8')
                and repair['fixed_full_probe_sha256'] == self.frozen_hashes['stage31/probe_full.py'],
                'admitted controller repair must change only the recorded metadata loop variable')
        expected = repair['old_controller_sources_sha256'] if self.version in ('001', '002') else repair['new_controller_sources_sha256']
        require(self.controllers == expected, 'archived controllers are not the admitted version for this round')
        for name, digest in expected.items():
            project = old_source if name == 'run_performance.py' and self.version in ('001', '002') else SOURCE / name
            require(sha(project) == digest, 'admitted controller bytes changed: ' + name)
        archived_repair = self.base / 'controller-repair.json'
        if self.version not in ('001', '002'):
            require(archived_repair.is_file() and sha(archived_repair) == sha(repair_path),
                    'repaired round must archive the explicit metadata-only controller admission')
        require(sha(SOURCE / 'contracts/main.json') == sha(self.contract_path), 'project/deployed contract differs')
        original = read(SEALED / 'contracts/main.json')
        require(self.contract['model'] == original['model'] and self.contract['correctness'] == original['correctness']
                and self.contract['candidate_sha256'] == original['candidate_sha256']
                and self.contract['training_argv'] == [x for x in original['training_argv']
                                                     if x != '--no-masked-softmax-fusion']
                and self.contract['model']['parameters'] == self.parameters
                and self.contract['stage31']['source_contract_sha256'] == sha(SEALED / 'contracts/main.json')
                and self.contract['stage31']['frozen_swiglu_candidate_sha256'] == SWIGLU_SHA,
                'only enabling default native fusion is permitted for the original model contract')
        require(self.manifest['commit'] == self.contract['megatron_commit'] == COMMIT
                and len(self.manifest['files']) == 600, 'all600 fixed native source identities required')
        perf = self.contract['performance']
        require(perf['steps'] == 120 and perf['independent_seeds'] == SEEDS
                and perf['arms'] == ['native', 'control', 'optimized'] and perf['process_orders'] == ORDERS
                and perf['tensor_snapshotting'] is False, 'original formal timer/rotation differs')
        scope = self.contract['attention']
        require(scope['shape'] == [128, 2, 8, 64] and scope['strides'] == [3072, 1536, 192, 1]
                and scope['storage_offsets'] == [0, 64, 128] and scope['scale'] == .125
                and scope['correctness']['audit_seed'] == 31101, 'fixed Attention scope differs')
        local = SOURCE / 'candidates' / self.version
        generation_path = local / 'generation.json'
        generation = read(generation_path)
        require(generation['model'] == 'Claude-Opus-4.8' and generation['claude_code'] is True
                and generation['sha256'] == self.candidate_sha == sha(local / 'candidate.py'),
                'generated candidate bytes/model/CC provenance differs')
        response = (local / generation['raw_response']).resolve()
        require(response.is_relative_to((BF / 'evidence').resolve())
                and sha(response) == generation['raw_response_sha256'], 'CC response path/hash differs')
        cc = read(response)
        require(cc['type'] == 'result' and cc['subtype'] == 'success' and cc['is_error'] is False
                and cc['session_id'] == generation['session_id'] and cc['num_turns'] == generation['num_turns'],
                'CC generation summary contradicts original service receipt')
        archived_generation = self.base / ('generation-' + self.version + '.json')
        if archived_generation.exists():
            require(sha(archived_generation) == sha(generation_path), 'archived generation receipt differs')
        self.generation_info = {'sha256': sha(generation_path), 'raw_response_sha256': sha(response),
                                'prompt_sha256': generation['prompt_sha256'], 'model': generation['model'],
                                'claude_code': True, 'session_id': generation['session_id']}
        backend_path = self.raw('backend-default.json')
        backend = read(backend_path)
        native_sources = {name: self.manifest['files'][name] for name in
                          ('megatron/core/transformer/dot_product_attention.py',
                           'megatron/core/fusions/fused_softmax.py', 'megatron/core/transformer/utils.py')}
        require(backend['status'] == 'passed' and backend['requested_backend'] == 'default-fusion'
                and backend['sources_unchanged'] is True and backend['megatron_commit'] == COMMIT
                and backend['source_manifest_sha256'] == sha(FORMAL / 'source_manifest.json')
                and backend['controller_sha256'] == sha(self.raw('probe_native_backend.py'))
                == sha(SOURCE / 'probe_native_backend.py') and backend['native_source_sha256'] == native_sources
                and len(backend['cases']) == 3 and {row['sequence'] for row in backend['cases']} == {128, 256, 512},
                'strong original default fused-backend source/shape evidence differs')
        extensions = {}
        for row in backend['cases']:
            require(row['status'] == 'passed' and row['route'] == 'cuda-extension'
                    and all(row[key] is True for key in
                            ('kernel_available', 'forward_completed', 'backward_completed',
                             'cuda_extension_forward_completed', 'cuda_extension_backward_completed',
                             'scaled_masked_softmax_fusion', 'finite_output', 'finite_gradient', 'mask_unchanged'))
                    and row['softmax_in_fp32'] is False and row['additional_softmax_scale'] is None
                    and row['softmax_offset'] is None and row['input_dtype'] == 'torch.bfloat16',
                    'default backend did not prove original extension forward/backward')
            for name in ('scaled_masked_softmax_cuda', 'scaled_upper_triang_masked_softmax_cuda'):
                item = row['extensions_after'][name]
                require(item['loaded'] is item['importable'] is True, 'native softmax binary unavailable')
                value = {'path': item['path'], 'sha256': item['sha256']}
                require(extensions.setdefault(name, value) == value, 'native binary differs between shapes')
        self.backend = {'path': REMOTE + '/backend-default.json', 'sha256': sha(backend_path),
                        'native_source_sha256': native_sources, 'native_extensions': extensions}
        capture = read(self.raw('native-default-capture-02/attention.json'))
        require(capture['calls'] == 4 and capture['capture_written'] is True
                and capture['native_source_sha256'] == native_sources
                and set(capture['raw_files']) == {'inputs.pt', 'output.pt', 'upstream-grad.pt',
                                                 'query-grad.pt', 'key-grad.pt', 'value-grad.pt'},
                'original native captured reference scope differs')
        self.capture = capture
        for name, row in capture['raw_files'].items():
            rel = 'native-default-capture-02/attention/' + name
            self.snapshot_records[rel] = row
            path = self.raw_snapshot(rel)
            if path.is_file():
                self.byte_check(path, row)
        probe_path = self.raw(self.args.probe or ('probe-full-' + self.version + '.json'))
        probe = read(probe_path)
        require(probe['schema'] == 'native-fused-full-attention-independent-probe-v1'
                and probe['status'] == 'passed' and probe['version'] == self.version
                and probe['candidate_sha256'] == self.candidate_sha
                and probe['verifier_sha256'] == sha(self.raw('probe_full.py'))
                == self.frozen_hashes['stage31/probe_full.py']
                and probe.get('verifier_path', REMOTE + '/probe_full.py') == REMOTE + '/probe_full.py'
                and probe['native_source_sha256'] == native_sources
                and probe['backend_proof_sha256'] == sha(backend_path)
                and probe['atol'] == 3e-4 and probe['rtol'] == 1e-3,
                'same frozen full Attention probe/candidate/tolerance/backend identity required')
        require(len(probe['checks']) >= 21
                and {tuple(row['shape']) for row in probe['checks']} >= {(128, 2, 8, 64), (256, 2, 8, 64), (512, 2, 8, 64)}
                and {row['case'] for row in probe['guard_checks']} >= {'cpu', 'float32'}
                and all(row['passed'] is True for row in probe['guard_checks']), 'probe full shape/guard coverage differs')
        for row in probe['checks']:
            require(row['passed'] is True and row['inputs_unchanged'] is True
                    and set(row['checks']) == {'output', 'query_grad', 'key_grad', 'value_grad'},
                    'complete full Attention output/QKV-gradient/immutability evidence required')
            for check in row['checks'].values():
                require(check['passed'] is True and check['bad_elements'] == 0
                        and math.isfinite(check['max_abs']) and check['max_abs'] >= 0,
                        'probe subcheck contradicts numerical pass')
        require(set(probe['native_capture_replay']) == {'output', 'query_grad', 'key_grad', 'value_grad'}
                and all(row['passed'] is True and row['exact_equal'] is True and row['bad_elements'] == 0
                        for row in probe['native_capture_replay'].values()), 'native captured replay is not exact')
        for row in probe['timing']:
            require(row['mode'] in ('forward', 'forward-backward') and set(row['raw_pairs']) == {'native', 'candidate'},
                    'independent Event microtiming scope differs')
            for arm in ('native', 'candidate'):
                values = row['raw_pairs'][arm]
                require(len(values) == 12 and all(math.isfinite(value) and value > 0 for value in values),
                        'invalid Event raw samples')
                close(row[arm + '_median_ms'], statistics.median(values), 'Event median differs from raw samples')
        self.probe_info = {'checks': len(probe['checks']), 'guards': len(probe['guard_checks']),
                           'sha256': sha(probe_path), 'verifier_sha256': probe['verifier_sha256']}
        self.provenance = {'frozen_sources_sha256': self.frozen_hashes, 'controller_sources_sha256': self.controllers,
                           'backend': self.backend, 'probe_path': REMOTE + '/' + probe_path.name,
                           'probe_sha256': sha(probe_path), 'verifier_path': REMOTE + '/probe_full.py',
                           'verifier_sha256': probe['verifier_sha256'],
                           'native_capture_receipt_sha256': sha(self.raw('native-default-capture-02/attention.json')),
                           'native_capture_raw_files': capture['raw_files'], 'candidate_sha256': self.candidate_sha,
                           'swiglu_candidate_sha256': SWIGLU_SHA, 'native_source_commit': COMMIT,
                           'native_source_file_count': 600,
                           'adaptation': 'Only outer launcher identity and controlled-left candidate identity gates '
                                         'adapted. Original model/source/all-tensor/master/precision/timer gates retained.'}

    def backup(self):
        path = self.base / ('raw-inventory-' + self.version + '.json')
        receipt_path = self.base / ('raw-backup-' + self.version + '-receipt.json')
        require(not receipt_path.exists() or path.exists(), 'backup receipt lacks inventory')
        if not path.exists():
            return
        rows = read(path)
        require(isinstance(rows, list) and bool(rows), 'nonempty raw inventory required')
        unique, present, missing = {}, 0, 0
        folders = {'main-' + arm + '-audit-' + self.version + '-' + self.args.audit_suffix
                   for arm in ('native', 'control', 'optimized')}
        for row in rows:
            rel = relative(row['path'])
            require(len(rel.parts) >= 3 and ((rel.parts[0] in folders and rel.parts[1] == 'snapshots')
                    or (rel.parts[:2] == ('native-default-capture-02', 'attention') and rel.suffix == '.pt'))
                    and row['bytes'] >= 0 and row['path'] not in self.inventory
                    and len(row['sha256']) == 64, 'raw inventory escaped scope or duplicates a path')
            self.inventory[row['path']] = row
            if row['path'] in self.snapshot_records:
                original = self.snapshot_records[row['path']]
                require(original['bytes'] == row['bytes'] and original['sha256'] == row['sha256'],
                        'raw capture inventory contradicts original native receipt')
            require(unique.setdefault(row['sha256'], row)['bytes'] == row['bytes'], 'same SHA has different size')
            raw = self.raw_snapshot(row['path'])
            if raw.is_file():
                self.byte_check(raw, row)
                object_path = self.base / 'raw-backup/objects' / row['sha256']
                if receipt_path.exists():
                    require(object_path.is_file() and raw.samefile(object_path), 'raw path is not a deduplicated hardlink')
                present += 1
            else:
                missing += 1
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt['status'] == 'verified' and receipt['version'] == self.version
                    and receipt['audit_suffix'] == self.args.audit_suffix and receipt['remote_root'] == REMOTE
                    and receipt['remote_deleted'] is False and receipt['inventory_sha256'] == sha(path)
                    and receipt['files'] == rows and receipt['verified_files'] == len(rows)
                    and receipt['raw_bytes'] == sum(row['bytes'] for row in rows)
                    and receipt['unique_files'] == len(unique)
                    and receipt['unique_bytes'] == sum(row['bytes'] for row in unique.values())
                    and receipt['deduplication'] == 'shared SHA256 objects and hardlinks'
                    and receipt['transfer_archive_retained'] is False
                    and receipt['backup_script_sha256'] == sha(SOURCE / 'backup_raw.py')
                    and receipt['gpu_timing_finished_acknowledged'] is True, 'raw backup receipt contradicts inventory')
        self.backup_info = {'inventory_available': True, 'receipt_available': receipt_path.exists(),
                            'inventory_sha256': sha(path), 'inventory_files': len(rows),
                            'files_available': present, 'files_missing': missing, 'unique_contents': len(unique)}

    def expected_command(self, folder, arm, steps, seed, python):
        root = REMOTE + '/' + folder.name
        result = [python, '-u', REMOTE + '/launch_attention.py', '--attention-receipt', root + '/attention.json',
                  '--delegate-launcher', REMOTE_SEALED + '/launch_swiglu-03.py',
                  '--swiglu-receipt', root + '/swiglu.json', '--gradnorm-launcher',
                  REMOTE_GRAD + '/launch_gradnorm-04.py', '--bf16-root', REMOTE_BF,
                  '--gradnorm-receipt', root + '/gradnorm.json', '--megatron-root', REMOTE_NATIVE,
                  '--contract', REMOTE + '/contracts/main.json', '--entry-evidence', root + '/entry.json']
        if arm != 'native':
            result += ['--candidate', REMOTE_BF + '/candidates/0008/candidate.py',
                       '--gradnorm-candidate', REMOTE_GRAD + '/candidate.py',
                       '--swiglu-candidate', REMOTE_SEALED + '/candidate-004.py']
        if arm == 'optimized':
            result += ['--attention-candidate', REMOTE + '/candidate-' + self.version + '.py']
        if steps == 3:
            result += ['--audit-dir', root + '/snapshots']
        result += self.contract['training_argv'] + self.training_suffix(folder, steps, seed)
        return result

    def training_suffix(self, folder, steps, seed):
        result = ['--train-iters', str(steps), '--seed', str(seed), '--log-interval', '1' if steps == 3 else '10']
        if steps == 3:
            result += ['--save', REMOTE + '/' + folder.name + '/snapshots', '--save-interval', '3',
                       '--save-wgrads-interval', '1', '--save-params-interval', '1']
        return result

    def entry(self, folder, arm, steps, seed):
        command, entry = read(folder / 'command.json'), read(folder / 'entry.json')
        require(command['case'] == folder.name and command['stage31_arm'] == arm
                and command['steps'] == steps and command['seed'] == seed and command['exit_code'] == 0
                and command['audit'] is (steps == 3) and command['profile'] is False
                and command['mode'] == ('native' if arm == 'native' else 'optimized')
                and command['contract_sha256'] == entry['contract_sha256'] == sha(self.contract_path)
                and command['controller_sources_sha256'] == self.controllers
                and command['wrapper_sha256'] == self.controllers['launch_attention.py']
                and command['attention_candidate_sha256'] == (self.candidate_sha if arm == 'optimized' else None)
                and command['swiglu_candidate_sha256'] == (SWIGLU_SHA if arm != 'native' else None)
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
        expected = {'parameters': 17895936, 'layers': 4, 'hidden_size': 512, 'vocab_size': 1024,
                    'bf16': True, 'fp16': False, 'class': 'megatron.core.models.gpt.gpt_model.GPTModel'}
        require(len(entry['models']) == 1 and all(entry['models'][0].get(key) == value for key, value in expected.items()),
                'actual built model differs')
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
                                             'residual_native_calls': 4 * steps, 'fallback_reasons': {}},
                    'CE source/device/coverage differs')
        require(datetime.fromisoformat(command['started_utc']) < datetime.fromisoformat(command['finished_utc'])
                and math.isfinite(command['whole_process_elapsed_seconds']) and command['whole_process_elapsed_seconds'] > 0,
                'invalid process chronology')
        return command, entry

    def hooks(self, folder, arm, steps):
        attention, activation, grad = (read(folder / name) for name in ('attention.json', 'swiglu.json', 'gradnorm.json'))
        enabled, controlled, observe, calls = arm == 'optimized', arm != 'native', steps == 3, 4 * steps
        require(attention['schema'] == 'native-complete-attention-dispatch-v1' and attention['enabled'] is enabled
                and attention['calls'] == calls and attention['candidate_calls'] == (calls if enabled else 0)
                and attention['native_calls'] == (0 if enabled else calls)
                and attention['fallback_calls'] == attention['unsupported_calls'] == 0
                and attention['native_routes_observed'] is observe
                and attention['native_fused_forward_calls'] == (calls if observe and not enabled else 0)
                and attention['native_fused_backward_calls'] == (calls if observe and not enabled else 0)
                and attention['native_torch_forward_calls'] == 0
                and attention['wrapper_sha256'] == self.controllers['launch_attention.py']
                and attention['delegate_launcher_sha256'] == sha(SEALED / 'launch_swiglu-03.py')
                and attention['candidate_sha256'] == (self.candidate_sha if enabled else None)
                and attention['native_source_sha256'] == self.backend['native_source_sha256']
                and attention['capture_written'] is False, 'Attention source/coverage/fair audit-only route counters differ')
        require(len(attention['configurations']) == 4
                and all(row == self.constants['ATTENTION_CONFIG'] for row in attention['configurations']),
                'actual default native Attention configuration differs')
        signatures = [((128, 2, 8, 64), (3072, 1536, 192, 1), 'torch.bfloat16', offset, True, True)
                      for offset in (0, 64, 128)]
        signatures.append(((2, 1, 128, 128), (16384, 16384, 128, 1), 'torch.bool', 0, False, True))
        require(attention['shapes'] == {str(tuple(signatures)): calls}, 'QKV layout/offset/mask coverage differs')
        for name, value in self.backend['native_extensions'].items():
            if name == 'scaled_masked_softmax_cuda' or not enabled:
                require(attention['native_extensions'].get(name) == value, 'formal strong default native binary differs')
        swiglu_sources = {name: self.manifest['files'][name] for name in
                         ('megatron/core/transformer/mlp.py', 'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
        require(activation['enabled'] is controlled and activation['calls'] == (calls if controlled else 0)
                and activation['native_calls'] == activation['fallback_calls'] == 0
                and activation['wrapper_sha256'] == sha(SEALED / 'launch_swiglu-03.py')
                and activation['candidate_sha256'] == (SWIGLU_SHA if controlled else None)
                and activation['native_source_sha256'] == swiglu_sources and activation['native_compiled_callable'] is True
                and activation.get('extension') is None and not activation.get('capture_written', False),
                'same native/default compiled or frozen SwiGLU004 scope differs')
        key = str(((128, 2, 4096), (8192, 4096, 1), 'torch.bfloat16'))
        require(activation['shapes'] == ({key: calls} if controlled else {}), 'SwiGLU004 call/shape coverage differs')
        if controlled or steps == 3:
            require(activation['configuration'] == self.constants['SWIGLU_CONFIG'], 'default SwiGLU configuration differs')
        require(grad['enabled'] is controlled and grad['calls'] == (steps if controlled else 0)
                and grad['fallback_calls'] == 0 and grad['wrapper_sha256'] == sha(self.frozen('gradnorm/launch_gradnorm-04.py'))
                and grad['candidate_sha256'] == (sha(self.frozen('gradnorm/candidate.py')) if controlled else None)
                and len(grad['norm_audit']) == (3 if controlled and observe else 0)
                and all(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm'] for row in grad['norm_audit']),
                'gradnorm001 source/call/numeric or untimed instrumentation differs')
        if steps == 120:
            require(not any((folder / name).exists() for name in ('snapshots', 'profile', 'activation', 'attention')),
                    'diagnostic artifacts in timed case')
        return {'attention': attention, 'swiglu': activation, 'gradnorm': grad,
                'raw_sha256': {name: sha(folder / name) for name in ('attention.json', 'swiglu.json', 'gradnorm.json')}}

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
        path = self.raw('audit-summary-main-' + self.version + '-' + suffix + '.json')
        audit = read(path)
        require(audit['schema'] == 'native-full-attention-stage31-three-way-audit-v1'
                and audit['status'] == 'completed' and audit['passed'] is True
                and audit['version'] == self.version and audit['model_name'] == 'main' and audit['suffix'] == suffix
                and audit['audit_seed'] == 31101 and audit['candidate_sha256'] == self.candidate_sha
                and audit['contract_sha256'] == sha(self.contract_path)
                and audit['wrapper_sha256'] == self.controllers['launch_attention.py']
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
            require(folder.parent == self.base and folder.name == 'main-' + arm + '-audit-' + self.version + '-' + suffix,
                    'unexpected numerical case root/name')
            command, entry = self.entry(folder, arm, 3, 31101)
            hooks = self.hooks(folder, arm, 3)
            require(item['hooks'] == hooks, 'raw numerical hook receipts differ from audit summary')
            folders[arm], commands[arm] = folder, command
            observation = hooks['attention']
            routes[arm] = {'case': folder.name, 'attention_receipt_sha256': hooks['raw_sha256']['attention.json'],
                           **{name: observation[name] for name in
                              ('native_routes_observed', 'native_fused_forward_calls', 'native_fused_backward_calls',
                               'native_torch_forward_calls', 'candidate_calls', 'fallback_calls')}}
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
            require(proof['stage31_comparison'] == label and proof['model_name'] == 'main' and proof['version'] == self.version
                    and proof['native_case'] == REMOTE + '/' + folders[left].name
                    and proof['optimized_case'] == REMOTE + '/' + folders[right].name
                    and proof['steps'] == 3 and proof['atol'] == 3e-4 and proof['rtol'] == 1e-3
                    and proof['comparison_device'] == 'CPU float64 copies'
                    and proof['comparator_sha256'] == audit['frozen_comparator_sha256']
                    and proof['stage31_runner_sha256'] == self.controllers['run_audit.py']
                    and proof['stage31_controller_sources_sha256'] == self.controllers
                    and proof['stage31_probe_sha256'] == self.probe_info['sha256']
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

    def performance(self, audit_path, audit_routes, *, diagnose_metadata_failure=False):
        path = self.base / ('performance-main-' + self.version + '-' + self.args.run_suffix + '.json')
        if not path.exists():
            return {'available': False, 'promoted': None, 'timed_processes': 0}
        perf = read(path)
        audit_link_ok = self.recorded(perf['audit_path']) == audit_path and perf['audit_sha256'] == sha(audit_path)
        if diagnose_metadata_failure:
            last_path, last_row = next(reversed(self.input_rows.items()))
            require(self.version == '002' and not audit_link_ok
                    and perf['audit_path'] == last_path and perf['audit_sha256'] == last_row['sha256']
                    and perf['promoted'] is False and perf['decision'] == 'not_promoted',
                    'only immutable nonpromoted002 with the exact known variable-shadow metadata bug may be diagnosed')
        else:
            require(audit_link_ok, 'performance numerical audit-summary path/SHA binding differs')
        require(perf['schema'] == 'native-full-attention-stage31-three-arm-performance-v1'
                and perf['status'] == 'validated' and perf['version'] == self.version
                and perf['model_name'] == 'main' and perf['run_suffix'] == self.args.run_suffix
                and perf['candidate_sha256'] == self.candidate_sha and perf['contract_sha256'] == sha(self.contract_path)
                and perf['wrapper_sha256'] == self.controllers['launch_attention.py']
                and perf['runner_sha256'] == self.controllers['run_performance.py']
                and perf['controller_sources_sha256'] == self.controllers and perf['provenance'] == self.provenance
                and perf['ce_candidate_sha256'] == self.contract['candidate_sha256']
                and perf['gradnorm_candidate_sha256'] == sha(self.frozen('gradnorm/candidate.py'))
                and perf['swiglu_candidate_sha256'] == SWIGLU_SHA and perf['model'] == self.contract['model']
                and perf['performance_native_routes_observed_per_call'] is False
                and perf['untimed_audit_route_provenance'] == audit_routes
                and perf['numerical_input_files_hashed_before_timing'] == self.input_count,
                'performance source/probe/model/audit binding or route-observation fairness differs')
        parser, validator, hashes = self.load_validator()
        require(all(perf['validator_adaptation'][key] == value for key, value in hashes.items()),
                'original in-memory outer-argv-only adaptation differs')
        require(perf['seeds'] == SEEDS and perf['orders'] == ORDERS
                and len(perf['pairs']) == 3 and len(perf['completed_cases']) == 9, 'complete declared nine-process rotation required')
        ratios = {name: [] for name in ('vs_native', 'vs_control', 'control_vs_native')}
        completed, previous_finish = [], None
        for pair, seed, order in zip(perf['pairs'], SEEDS, ORDERS):
            require(pair['seed'] == seed and pair['order'] == order and set(pair['arms']) == {'native', 'control', 'optimized'},
                    'per-seed three-arm rotation differs')
            means, models = {}, {}
            for arm in order:
                folder = self.base / ('main-' + arm + '-perf-' + self.version + '-' + self.args.run_suffix + '-' + str(seed))
                command, entry = self.entry(folder, arm, 120, seed)
                hooks = self.hooks(folder, arm, 120)
                validator.EXPECTED_COMMANDS[str(folder)] = self.expected_command(folder, arm, 120, seed, command['command'][0])
                fresh = validator.validate_case(folder, command['mode'], seed, self.contract,
                                                sha(self.contract_path), self.contract['candidate_sha256'])
                fresh.update(stage31_arm=arm, hooks=hooks)
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
                'native_route_observed_per_timed_call': False, 'promoted': promoted,
                'audit_metadata_binding_passed': audit_link_ok,
                'diagnostic_only': diagnose_metadata_failure, **results}


def verify(args):
    archive = Archive(args)
    archive.foundations()
    archive.backup()
    path, comparisons, routes = archive.audit()
    performance = archive.performance(path, routes)
    return {'schema': 'native-full-attention-stage31-cpu-archive-verification-v1', 'status': 'verified',
            'version': args.version, 'candidate_sha256': archive.candidate_sha, 'contract_sha256': sha(archive.contract_path),
            'controller_sources_sha256': archive.controllers, 'frozen_sources_checked': len(archive.frozen_hashes),
            'verifier_sha256': sha(__file__), 'probe': archive.probe_info, 'generation': archive.generation_info,
            'audit_sha256': sha(path), 'comparisons': comparisons, 'untimed_audit_routes': routes, 'performance': performance,
            'snapshot_inventory_records': len(archive.snapshot_records), 'snapshot_files_available': archive.snapshot_present,
            'snapshot_files_missing': archive.snapshot_missing, 'raw_backup': archive.backup_info,
            'native_extension_binary_bytes_verified_locally': False,
            'gpu_rerun': False, 'tensor_arithmetic_replayed': False,
            'scope': 'All recorded source/contract/controller/probe/generation identities, three-way637-tensor/master '
                     'receipts and every original training interval log rechecked on CPU. Available raw tensors are '
                     'SHA256 checked; missing raw bytes remain explicit. Native extension hashes are bound to '
                     'untimed backend/audit and actual formal receipts; binary bytes are not asserted present locally. '
                     'No Torch/candidate import, GPU rerun, tensor arithmetic replay or production extrapolation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--base', type=Path, default=SOURCE / 'evidence')
    parser.add_argument('--probe', help='Relative probe JSON name; default probe-full-NNN.json')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    try:
        require(args.version.isdigit() and len(args.version) == 3
                and all(value.replace('-', '').isalnum() for value in (args.audit_suffix, args.run_suffix)),
                'three-digit candidate and simple immutable suffixes required')
        result = verify(args)
    except Exception as error:
        result = {'status': 'rejected', 'version': args.version, 'gpu_rerun': False,
                  'error_type': type(error).__name__, 'error': str(error)}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
