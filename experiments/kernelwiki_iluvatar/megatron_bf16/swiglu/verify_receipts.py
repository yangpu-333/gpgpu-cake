"""CPU archive verification of default SwiGLU probes, audits and raw timing.

No Torch, candidate or experiment controller is imported, and no GPU work is
repeated. Numerical JSON receipts are checked element by element. Available
snapshot/binary files are hashed; missing raw bytes are reported explicitly.
The original standard-library interval parser and frozen performance validator
are the only imported experiment modules.
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import sys


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'swiglu'
FORMAL = ROOT.parent / 'megatron_training_entry'
DEFAULT_BASE = ROOT / 'evidence/swiglu-stage29'
REMOTE = '/tmp/kda-swiglu-stage29'
REMOTE_BF = '/tmp/kda-bf16-continuation/megatron_bf16'
REMOTE_NATIVE = '/tmp/kda-bf16-continuation/megatron-native-5be9626'
REMOTE_GRAD = '/tmp/kda-gradnorm-stage26'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
ORDERS = (['native', 'control', 'optimized'], ['optimized', 'control', 'native'],
          ['control', 'native', 'optimized'])
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
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right, name):
    require(isinstance(left, (int, float)) and isinstance(right, (int, float))
            and math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), name)


class Archive:
    def __init__(self, base, version):
        self.base, self.version = Path(base).resolve(), version
        self.candidate = self.base / ('candidate-' + version + '.py')
        self.contract_path = self.base / 'contracts/main.json'
        self.contract = read(self.contract_path)
        self.manifest = read(FORMAL / 'source_manifest.json')
        self.extension = None
        self.snapshot_present = self.snapshot_missing = 0
        self.snapshot_seen = set()
        self.binary_bytes_checked = False
        self.controllers = {}
        self.build_receipt = None
        self.build_receipt_sha = None
        self.audit_wrapper = None
        self.raw_inventory = {}
        self.backup_summary = {'available': False, 'files_hashed': 0, 'files_not_present_locally': 0}

    def raw(self, name):
        path = self.base / name
        require(path.resolve().is_relative_to(self.base), 'archive path escapes evidence root')
        require(path.is_file(), 'missing raw archive file: ' + str(path))
        return path

    def recorded(self, value):
        value = str(value).replace('\\', '/')
        for remote, local in ((REMOTE + '/', self.base), (REMOTE_BF + '/', ROOT),
                              ('/tmp/kda-bf16-continuation/megatron_training_entry/', FORMAL)):
            if value.startswith(remote):
                relative = PurePosixPath(value[len(remote):])
                require('..' not in relative.parts and not relative.is_absolute(), 'unsafe recorded path')
                return local.joinpath(*relative.parts)
        if value == REMOTE_GRAD + '/candidate.py':
            return ROOT / 'gradnorm_candidates/001/candidate.py'
        if value == REMOTE_GRAD + '/launch_gradnorm-04.py':
            return ROOT / 'evidence/gradnorm-stage26/launch_gradnorm-04.py'
        raise ValueError('unmapped recorded path: ' + value)

    def bind_controller(self, name, expected):
        raw = self.raw(name)
        require(sha(raw) == expected, 'deployed controller SHA differs: ' + name)
        require(raw.read_text(encoding='utf-8') == (SOURCE / name).read_text(encoding='utf-8'),
                'project controller text differs from deployed archive: ' + name)
        self.controllers[name] = expected

    def controller_by_sha(self, names, expected):
        matches = [name for name in names if (self.base / name).is_file()
                   and sha(self.base / name) == expected]
        require(len(matches) == 1, 'exact archived controller identity required for SHA ' + expected)
        self.bind_controller(matches[0], expected)
        return self.raw(matches[0])

    def raw_byte_path(self, relative):
        relative = PurePosixPath(relative)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe raw backup path')
        choices = [self.base.joinpath(*relative.parts),
                   (self.base / 'raw-backup').joinpath(*relative.parts)]
        return next((path for path in choices if path.is_file()), choices[0])

    def backup(self):
        inventory_path = self.base / 'raw-inventory.json'
        if not inventory_path.is_file():
            return
        rows = read(inventory_path)
        require(isinstance(rows, list) and bool(rows), 'nonempty raw snapshot/binary inventory required')
        self.raw_inventory = {row['path']: row for row in rows}
        require(len(self.raw_inventory) == len(rows), 'raw backup inventory has duplicate paths')
        unique = {}
        hashed_inodes = {}
        present = missing = 0
        for row in rows:
            relative = PurePosixPath(row['path'])
            require(not relative.is_absolute() and '..' not in relative.parts and row['bytes'] > 0,
                    'safe finite raw inventory path/size required')
            previous = unique.setdefault(row['sha256'], row)
            require(previous['bytes'] == row['bytes'], 'same raw SHA has conflicting byte count')
            local = self.raw_byte_path(row['path'])
            if not local.is_file():
                missing += 1
                continue
            stat = local.stat()
            require(stat.st_size == row['bytes'], 'raw backup byte size differs: ' + row['path'])
            key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if key not in hashed_inodes:
                hashed_inodes[key] = sha(local)
            require(hashed_inodes[key] == row['sha256'], 'raw backup content differs: ' + row['path'])
            present += 1
        receipt_path = self.base / 'raw-backup-receipt.json'
        receipt_checked = False
        if receipt_path.is_file():
            receipt = read(receipt_path)
            require(receipt['status'] == 'verified' and receipt['remote_root'] == REMOTE
                    and receipt['remote_deleted'] is False and receipt['inventory_sha256'] == sha(inventory_path)
                    and receipt['files'] == rows and receipt['verified_files'] == len(rows)
                    and receipt['raw_bytes'] == sum(row['bytes'] for row in rows)
                    and receipt['unique_files'] == len(unique)
                    and receipt['unique_bytes'] == sum(row['bytes'] for row in unique.values()),
                    'raw backup receipt is inconsistent with full inventory')
            receipt_checked = True
        self.backup_summary = {'available': True, 'inventory_sha256': sha(inventory_path),
                               'receipt_checked': receipt_checked, 'inventory_files': len(rows),
                               'files_hashed': present, 'files_not_present_locally': missing,
                               'unique_contents': len(unique)}

    def foundations(self):
        contract = self.contract
        require(self.manifest['commit'] == contract['megatron_commit'] == COMMIT
                and len(self.manifest['files']) == 600, 'fixed complete 600-file source manifest required')
        require(contract['correctness']['atol'] == 3e-4 and contract['correctness']['rtol'] == 1e-3
                and contract['correctness']['steps'] == 3, 'frozen numerical tolerances differ')
        require(contract['model']['parameters'] == 17895936
                and contract['model']['num_layers'] == 4 and contract['model']['hidden_size'] == 512
                and contract['model']['ffn_hidden_size'] == 2048
                and contract['model']['sequence_length'] == 128
                and contract['model']['micro_batch_size'] == contract['model']['global_batch_size'] == 2
                and contract['model']['vocab_size'] == 1024
                and contract['model']['bf16'] is True
                and contract['model']['gated_linear_unit'] is True
                and contract['model']['bias_activation_fusion'] is True,
                'sealed native-default SwiGLU model differs')
        require('--swiglu' in contract['training_argv']
                and '--no-bias-swiglu-fusion' not in contract['training_argv']
                and contract['performance']['steps'] == 120
                and contract['performance']['independent_seeds'] == [26002, 26003, 26004],
                'native default fusion or declared training timer contract differs')
        require(sha(ROOT / 'candidates/0008/candidate.py') == contract['candidate_sha256'],
                'sealed CE0008 parent differs')
        require(self.contract_path.read_text(encoding='utf-8')
                == (SOURCE / 'contracts/main.json').read_text(encoding='utf-8'),
                'project contract differs from tested deployed contract')
        original = SOURCE / 'candidates' / self.version / 'candidate.py'
        require(original.read_text(encoding='utf-8') == self.candidate.read_text(encoding='utf-8'),
                'original model-produced candidate text differs from tested candidate')
        generation = read(original.parent / 'generation.json')
        generated_sha = generation.get('sha256', generation.get('sources', {}).get('candidate.py'))
        equivalent = {sha(original), sha(self.candidate),
                      hashlib.sha256(original.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
        require(generated_sha in equivalent and generation['model'] == 'Claude-Opus-4.8',
                'candidate generation identity or equivalent source SHA differs')

    def build(self, extension):
        if not extension.get('required', False):
            require(extension.get('passed', True) is True, 'non-extension numerical probe failed')
            return
        path = self.raw('cpp-build-' + self.version + '-receipt.json')
        build = read(path)
        require(build['status'] == 'built' and build['torch'] == '2.4.1' and build['cuda'] == '10.2',
                'compatible explicit extension build required')
        expected_sources = {}
        generation = read(SOURCE / 'candidates' / self.version / 'generation.json')
        for suffix in ('.cpp', '.cu'):
            name = 'swiglu-' + self.version + suffix
            actual = self.raw(name)
            original = SOURCE / 'candidates' / self.version / ('swiglu' + suffix)
            require(actual.read_text(encoding='utf-8') == original.read_text(encoding='utf-8'),
                    'compiled model-produced source differs: ' + name)
            expected_sources[name] = sha(actual)
            equivalents = {sha(actual), sha(original),
                           hashlib.sha256(original.read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
            require(generation['sources']['swiglu' + suffix] in equivalents,
                    'compiled source generation SHA differs: ' + name)
        build_script = PurePosixPath(build['build_script_path']).name
        require(build_script in ('build_cpp.py', 'build_cpp-02.py'), 'allowed explicit build controller required')
        self.bind_controller(build_script, build['build_script_sha256'])
        module = 'kda_swiglu_stage29_' + self.version
        artifact = REMOTE + '/cpp-build-' + self.version + '/' + module + '.so'
        require(build['name'] == module and build['artifact'] == artifact
                and build['build_directory'] == REMOTE + '/cpp-build-' + self.version
                and build['build_script_path'] == REMOTE + '/' + build_script
                and build['sources'] == extension['sources'] == expected_sources
                and extension['receipt_sha256'] == sha(path)
                and extension['artifact'] == artifact
                and extension['artifact_sha256'] == build['artifact_sha256']
                and extension['build_script_sha256'] == build['build_script_sha256']
                and extension['build_script_path'] == build['build_script_path']
                and extension['passed'] is True
                and extension['build_script_bytes_checked'] is True,
                'extension build/probe source or artifact receipt differs')
        expected_cuda_flags = ['-O3', '--fmad=false'] if build_script == 'build_cpp.py' else ['-O3', '-ffp-contract=off']
        require(build['extra_cflags'] == ['-O3', '-ffp-contract=off']
                and build['extra_cuda_cflags'] == expected_cuda_flags,
                'explicit non-contracting extension compiler flags differ')
        self.extension = {'path': artifact, 'sha256': build['artifact_sha256']}
        self.build_receipt, self.build_receipt_sha = build, sha(path)
        for binary in (self.raw_byte_path(artifact.removeprefix(REMOTE + '/')),
                       self.base / ('artifacts-' + self.version) / (module + '.so')):
            if binary.is_file():
                require(sha(binary) == build['artifact_sha256'], 'archived extension binary bytes differ')
                self.binary_bytes_checked = True
        receipt = self.base / ('artifacts-' + self.version) / 'receipt.json'
        if receipt.is_file():
            require(read(receipt)['artifact_sha256'] == build['artifact_sha256'],
                    'binary backup receipt differs from explicit build')

    def probe(self, path):
        proof = read(path)
        require(proof['status'] == 'passed' and proof['candidate_sha256'] == sha(self.candidate),
                'independent numerical probe candidate/status differs')
        v2 = proof.get('schema') in ('native-compiled-swiglu-independent-probe-v2',
                                   'native-compiled-swiglu-independent-probe-v3')
        names = [path.name for path in self.base.glob('probe_candidate*.py')
                 if (SOURCE / path.name).is_file()]
        self.controller_by_sha(names, proof['verifier_sha256'])
        records = proof['checks']
        require(len(records) == (37 if v2 else 22), 'complete fixed standalone numerical matrix required')
        for row in records:
            require(row['input_unchanged'] is True, 'standalone candidate mutated input')
            if v2:
                require(row['upstream_unchanged'] is True, 'standalone candidate mutated upstream gradient')
            require([item['name'] for item in row['checks']] == ['output', 'input_grad'],
                    'standalone forward and input-gradient coverage differs')
            for item in row['checks']:
                require(item['passed'] is True and math.isfinite(item['max_abs']) and item['max_abs'] >= 0
                        and item['unequal_elements'] >= 0
                        and item['bitwise_equal'] is (item['unequal_elements'] == 0),
                        'standalone numerical check metadata inconsistent')
                if v2:
                    require(all(item[key] is True for key in
                                ('shape_equal', 'dtype_equal', 'device_equal', 'finite')),
                            'standalone shape/dtype/device/finite check failed')
        require([item['name'] for item in proof['guard_checks']]
                == ['cpu', 'FP32', 'odd-lastdim', 'noncontiguous-input']
                and all(item['passed'] is True for item in proof['guard_checks']),
                'complete unsupported-input guard matrix required')
        if v2:
            groups = {name: sum(row['group'] == name for row in records) for name in
                      ('original', 'fresh-data-reuse', 'alternating-sizes', 'storage-offset', 'current-stream')}
            require(groups == {'original': 22, 'fresh-data-reuse': 6, 'alternating-sizes': 5,
                               'storage-offset': 2, 'current-stream': 2}
                    and proof['original_case_count'] == 22 and proof['total_case_count'] == 37,
                    'fresh-data/layout/current-stream standalone coverage differs')
            expected = {name: self.manifest['files'][name] for name in
                        ('megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
            require(proof['native_source_sha256'] == expected
                    and proof['torch'] == '2.4.1' and proof['gpu'] in ('BI-V150', 'Iluvatar BI-V150'),
                    'independent probe native source/GPU/runtime differs')
            self.build(proof['extension'])
        else:
            require(not (self.base / ('cpp-build-' + self.version + '-receipt.json')).exists(),
                    'compiled candidates require the v2 build/stream/aliasing numerical probe')
        require(len(proof['performance']) == 2
                and [row['backward'] for row in proof['performance']] == [False, True],
                'standalone forward and forward/backward Event measurements required')
        for row in proof['performance']:
            require(set(row['samples_us']) == {'native', 'candidate'}, 'Event arms differ')
            for arm, values in row['samples_us'].items():
                require(len(values) == 15 and all(math.isfinite(x) and x > 0 for x in values),
                        'complete finite paired Event samples required')
                close(row['median_us'][arm], statistics.median(values), 'Event sample median differs')
        require('torch.cuda.Event' in proof['timer'], 'standalone Event timing interface differs')
        return {'path': str(path), 'sha256': sha(path), 'cases': len(records),
                'numeric_checks': len(records) * 2, 'guard_checks': 4}

    def hooks(self, folder, arm, steps, wrapper):
        activation, grad = read(folder / 'swiglu.json'), read(folder / 'gradnorm.json')
        use_activation, use_grad = arm == 'optimized', arm != 'native'
        expected_source = {name: self.manifest['files'][name] for name in
                           ('megatron/core/transformer/mlp.py',
                            'megatron/core/fusions/fused_bias_swiglu.py', 'megatron/core/jit.py')}
        require(activation['enabled'] is use_activation
                and activation['calls'] == (4 * steps if use_activation else 0)
                and activation['fallback_calls'] == 0 and activation['wrapper_sha256'] == sha(wrapper)
                and activation['candidate_sha256'] == (sha(self.candidate) if use_activation else None)
                and activation['native_source_sha256'] == expected_source
                and activation['native_compiled_callable'] is True,
                'activation dispatch/default native source/artifact provenance differs')
        extension = activation.get('extension')
        if not use_activation or self.extension is None:
            require(extension is None, 'unexpected runtime compiled artifact receipt')
        else:
            require(extension['path'] == self.extension['path']
                    and extension['sha256'] == self.extension['sha256'],
                    'runtime compiled artifact differs from independent build probe')
            if 'sources' in extension:
                require(extension['sources'] == self.build_receipt['sources'],
                        'runtime compiled source hashes differ from explicit build')
            if 'build_receipt_sha256' in extension:
                require(extension['build_receipt_sha256'] == self.build_receipt_sha,
                        'runtime build receipt digest differs')
            if 'build_script_sha256' in extension:
                require(extension['build_script_sha256'] == self.build_receipt['build_script_sha256'],
                        'runtime build controller digest differs')
        if use_activation:
            key = str(((128, 2, 4096), (8192, 4096, 1), 'torch.bfloat16'))
            require(activation['native_calls'] == 0 and activation['shapes'] == {key: 4 * steps}
                    and activation['configuration'] == EXPECTED_CONFIG,
                    'actual optimized activation dtype/layout/default fusion configuration differs')
        if steps == 3:
            require(activation['configuration'] == EXPECTED_CONFIG, 'actual audit configuration differs')
        else:
            require(not activation.get('capture_written', False), 'tensor capture in timed process')
            if not use_activation:
                require(activation['native_calls'] == 0 and activation['shapes'] == {},
                        'native/control timed arm unexpectedly received candidate dispatch')
        grad_candidate = ROOT / 'gradnorm_candidates/001/candidate.py'
        grad_launcher = ROOT / 'evidence/gradnorm-stage26/launch_gradnorm-04.py'
        require(grad['enabled'] is use_grad and grad['calls'] == (steps if use_grad else 0)
                and grad['fallback_calls'] == 0 and grad['wrapper_sha256'] == sha(grad_launcher)
                and grad['candidate_sha256'] == (sha(grad_candidate) if use_grad else None),
                'gradnorm dispatch/source provenance differs')
        require(len(grad['norm_audit']) == (3 if use_grad and steps == 3 else 0),
                'gradnorm audit instrumentation scope differs')
        for row in grad['norm_audit']:
            require(row['bitwise_equal'] is True and row['native_norm'] == row['candidate_norm'],
                    'actual training norm bitwise check failed')
        return activation, grad

    def entry(self, folder, arm, steps, seed):
        command, entry = read(folder / 'command.json'), read(folder / 'entry.json')
        require(command['case'] == folder.name and command['steps'] == steps and command['seed'] == seed
                and command['mode'] == ('native' if arm == 'native' else 'optimized')
                and command['audit'] is (steps == 3) and command['profile'] is False
                and command['exit_code'] == 0 and entry['status'] == 'completed'
                and command['contract_sha256'] == entry['contract_sha256'] == sha(self.contract_path),
                'successful fixed-contract case metadata differs')
        integration = {name: sha(ROOT / name) for name in ('launch_training.py', 'kernel_adapter.py', 'run_case.py')}
        integration.update({'formal/' + name: sha(FORMAL / name)
                            for name in ('runtime_compat.py', 'source_manifest.json')})
        require(entry['native_sources_sha256'] == self.manifest['files']
                and entry['megatron_commit'] == COMMIT and entry['integration_sha256'] == integration
                and entry['launcher_sha256'] == integration['launch_training.py']
                and entry['formal_entry'] == REMOTE_NATIVE + '/pretrain_gpt.py'
                and entry['profile_hooks'] == [] and 'profiler_activities' not in entry,
                '600 native source/frozen launcher/original training entry gate differs')
        require(entry['runtime_compatibility']['torch'] == '2.4.1'
                and entry['runtime_compatibility']['optional_backends_disabled_in_process']
                == ['transformer_engine', 'apex'], 'shared native runtime compatibility differs')
        require(len(entry['models']) == 1, 'one original GPT model chunk required')
        model = entry['models'][0]
        expected = {'parameters': 17895936, 'layers': 4, 'hidden_size': 512, 'vocab_size': 1024,
                    'bf16': True, 'fp16': False,
                    'class': 'megatron.core.models.gpt.gpt_model.GPTModel'}
        require(all(model.get(k) == v for k, v in expected.items()), 'actual native GPT model differs')
        require(entry['iluvatar_kernels'] is (arm != 'native'), 'CE opt-in scope differs')
        if arm == 'native':
            require(entry['kernel_adapters'] == [], 'native installed a CE candidate')
        else:
            require(len(entry['kernel_adapters']) == 1, 'one CE adapter required')
            adapter = entry['kernel_adapters'][0]
            require(all(adapter[k] is True for k in ('enabled', 'active', 'provenance_verified'))
                    and adapter['candidate_sha256'] == adapter['candidate_expected_sha256']
                    == self.contract['candidate_sha256']
                    and adapter['runtime_world_size'] == 1
                    and adapter['verified_devices'] in ({'0': 'BI-V150'}, {'0': 'Iluvatar BI-V150'})
                    and adapter['counts'] == {'ce_candidate_calls': steps, 'ce_native_calls': 0,
                                              'residual_native_calls': 4 * steps, 'fallback_reasons': {}},
                    'actual CE candidate provenance/coverage/device differs')
        suffix = ['--train-iters', str(steps), '--seed', str(seed), '--log-interval', '1' if steps == 3 else '10']
        if steps == 3:
            suffix += ['--save', REMOTE + '/' + folder.name + '/snapshots', '--save-interval', '3',
                       '--save-wgrads-interval', '1', '--save-params-interval', '1']
        require(entry['training_argv'] == self.contract['training_argv'] + suffix,
                'actual formal training argv differs from contract')
        return command, entry

    def tensors(self, report):
        checks, tensors = report['checks'], report['tensor_comparisons']
        computed = {'total': len(checks), 'passed': sum(item['passed'] is True for item in checks),
                    'failed': sum(item['passed'] is not True for item in checks)}
        require(report['check_summary'] == computed and computed['failed'] == 0
                and report['passed'] is True and computed['total'] == 1475,
                'three-step summary differs from actual complete gate list')
        require(len(tensors) == 637 and all(item['passed'] is True for item in tensors),
                'complete fixed-model tensor comparison coverage required')
        groups = {}
        for item in tensors:
            groups[item['group']] = groups.get(item['group'], 0) + 1
            require(item['shape_dtype_layout_match'] is True and item['all_elements_finite'] is True
                    and item['native_shape'] == item['optimized_shape']
                    and item['native_dtype'] == item['optimized_dtype']
                    and item['native_numel'] == item['optimized_numel']
                    and item['elements_outside_tolerance'] == 0
                    and math.isfinite(item['max_tolerance_fraction'])
                    and item['max_tolerance_fraction'] <= 1
                    and item['exact_equal'] is (item['elements_not_exactly_equal'] == 0)
                    and (item['native_sha256'] != item['optimized_sha256'] or item['exact_equal'] is True),
                    'tensor comparison fields contradict numerical pass: ' + item['path'])
            require(not item['exact_required'] or item['exact_equal'] is True,
                    'required exact initialization/input/master consistency failed')
            require(item['atol'] == (0 if item['exact_required'] else 3e-4)
                    and item['rtol'] == (0 if item['exact_required'] else 1e-3),
                    'tensor comparison tolerance changed')
        require(groups == report['tensor_groups'], 'tensor group counts inconsistent')
        require(all(item['passed'] is True for item in report['raw_value_comparisons']),
                'recursive raw metadata comparison failed')
        for item in report['input_files']:
            local = self.recorded(item['path'])
            if '/snapshots/' in item['path']:
                relative = item['path'].removeprefix(REMOTE + '/')
                if relative in self.raw_inventory:
                    raw_item = self.raw_inventory[relative]
                    require(raw_item['bytes'] == item['bytes'] and raw_item['sha256'] == item['sha256'],
                            'numerical snapshot inventory differs from raw backup mapping')
                local = self.raw_byte_path(relative)
                if item['path'] in self.snapshot_seen:
                    continue
                self.snapshot_seen.add(item['path'])
                if local.is_file():
                    require(local.stat().st_size == item['bytes'] and sha(local) == item['sha256'],
                            'archived raw snapshot bytes differ')
                    self.snapshot_present += 1
                else:
                    self.snapshot_missing += 1
            else:
                require(local.is_file() and local.stat().st_size == item['bytes']
                        and sha(local) == item['sha256'], 'inventoried non-tensor evidence bytes differ')
        return {'checks': computed['total'], 'tensors': len(tensors),
                'all_tensors_exact_equal': all(item['exact_equal'] is True for item in tensors),
                'all_tensors_bitwise': all(item['native_sha256'] == item['optimized_sha256'] for item in tensors)}

    def audit(self, suffix):
        path = self.raw('audit-summary-main-' + self.version + '-' + suffix + '.json')
        summary = read(path)
        require(summary['schema'] == 'megatron-default-swiglu-three-way-audit-v1'
                and summary['version'] == self.version and summary['case_suffix'] == suffix
                and summary['passed'] is True and summary['swiglu_receipt_guard_passed'] is True
                and summary['candidate_sha256'] == sha(self.candidate)
                and summary['contract_sha256'] == sha(self.contract_path)
                and summary['frozen_comparator_sha256'] == sha(ROOT / 'compare_training.py'),
                'full numerical summary provenance differs')
        self.bind_controller('run_candidate_audit.py', summary['runner_sha256'])
        wrapper = self.controller_by_sha(('launch_swiglu-02.py', 'launch_swiglu-03.py'),
                                         summary['wrapper_sha256'])
        self.audit_wrapper = wrapper
        bound_probe = self.recorded(summary['probe_path'])
        require(sha(bound_probe) == summary['probe_sha256'], 'audit-bound independent probe bytes differ')
        self.probe(bound_probe)
        folders = {'native': self.base / 'main-native-audit',
                   'control': self.base / ('main-control-audit-' + self.version + '-' + suffix),
                   'optimized': self.base / ('main-optimized-audit-' + self.version + '-' + suffix)}
        models = {}
        for arm, folder in folders.items():
            _, entry = self.entry(folder, arm, 3, 29101)
            self.hooks(folder, arm, 3, self.raw('launch_swiglu.py') if arm == 'native' else wrapper)
            models[arm] = entry['models']
        require(models['native'] == models['control'] == models['optimized'],
                'three numerical arms constructed different models')
        expected = {'controlled': ('control', 'optimized'), 'native_optimized': ('native', 'optimized'),
                    'native_control': ('native', 'control')}
        require(set(summary['comparisons']) == set(expected), 'three numerical comparisons required')
        results = {}
        for label, (left, right) in expected.items():
            item = summary['comparisons'][label]
            raw = self.recorded(item['path'])
            proof = read(raw)
            require(raw.parent == self.base and sha(raw) == item['sha256']
                    and item['passed'] is True and proof['stage29_comparison'] == label
                    and proof['swiglu_receipt_guard_passed'] is True
                    and proof['stage29_runner_sha256'] == summary['runner_sha256']
                    and proof['comparator_sha256'] == summary['frozen_comparator_sha256']
                    and proof['swiglu_candidate_sha256'] == sha(self.candidate)
                    and proof['native_case'] == REMOTE + '/' + folders[left].name
                    and proof['optimized_case'] == REMOTE + '/' + folders[right].name
                    and proof['steps'] == 3 and proof['atol'] == 3e-4 and proof['rtol'] == 1e-3,
                    'numerical comparison source/case/tolerance differs')
            results[label] = self.tensors(proof)
            require(item['check_summary'] == proof['check_summary']
                    and item['tensor_comparisons'] == results[label]['tensors']
                    and item['all_tensors_bitwise_equal'] is results[label]['all_tensors_exact_equal'],
                    'audit summary does not reflect complete tensor records')
            if label == 'controlled':
                guards = [check for check in proof['checks'] if 'optimized_receipt' in check]
                require(len(guards) == 1
                        and guards[0]['optimized_receipt'] == read(folders['optimized'] / 'swiglu.json')
                        and guards[0]['control_receipt'] == read(folders['control'] / 'swiglu.json'),
                        'controlled numerical guard differs from raw runtime receipts')
        return path, summary, results

    def performance(self, suffix, audit_path, audit_summary):
        path = self.base / ('performance-main-' + self.version + '-' + suffix + '.json')
        if not path.is_file():
            return {'available': False, 'promoted': None, 'timed_processes': 0}
        perf = read(path)
        require(perf['schema'] == 'megatron-default-swiglu-three-arm-performance-v1'
                and perf['status'] == 'validated' and perf['version'] == self.version
                and perf['candidate_sha256'] == sha(self.candidate)
                and perf['contract_sha256'] == sha(self.contract_path)
                and perf['audit_sha256'] == sha(audit_path)
                and self.recorded(perf['audit_path']) == audit_path
                and perf['probe_sha256'] == audit_summary['probe_sha256']
                and perf['ce_candidate_sha256'] == self.contract['candidate_sha256']
                and perf['gradnorm_candidate_sha256'] == sha(ROOT / 'gradnorm_candidates/001/candidate.py')
                and perf['model'] == self.contract['model'],
                'performance source/numerical prerequisite provenance differs')
        self.bind_controller('run_performance.py', perf['runner_sha256'])
        wrapper = self.audit_wrapper
        require(perf['wrapper_sha256'] == sha(wrapper), 'performance activation wrapper differs')
        audit_optimized = self.base / ('main-optimized-audit-' + self.version + '-' + audit_summary['case_suffix'])
        require(perf['audited_extension'] == read(audit_optimized / 'swiglu.json').get('extension'),
                'timed extension receipt differs from full numerical audit')
        parser_spec = importlib.util.spec_from_file_location('swiglu_archive_native_parser',
                                                           FORMAL / 'summarize_performance.py')
        parser = importlib.util.module_from_spec(parser_spec)
        parser_spec.loader.exec_module(parser)
        frozen_source = (ROOT / 'summarize_performance.py').read_text(encoding='utf-8')
        start = frozen_source.index("    invocation=command['command']")
        end = frozen_source.index("    runtime=entry['runtime_compatibility']", start)
        replacement = """    invocation=command['command']
    require(invocation==EXPECTED_COMMANDS[str(folder)],'exact three-arm default SwiGLU outer wrapper invocation')
    require(entry['formal_entry']==EXPECTED_FORMAL_ENTRY,'original pinned formal pretrain entry')
"""
        adapted = frozen_source[:start] + replacement + frozen_source[end:]
        hashes = {'frozen_validator_sha256': sha(ROOT / 'summarize_performance.py'),
                  'adapted_validator_sha256': hashlib.sha256(adapted.encode()).hexdigest(),
                  'replaced_original_argv_block_sha256': hashlib.sha256(frozen_source[start:end].encode()).hexdigest(),
                  'replacement_argv_block_sha256': hashlib.sha256(replacement.encode()).hexdigest()}
        require(all(perf['validator_adaptation'][key] == value for key, value in hashes.items()),
                'frozen in-memory outer-argv validator adaptation differs')
        validator_spec = importlib.util.spec_from_file_location('swiglu_archive_frozen_validator',
                                                              ROOT / 'summarize_performance.py')
        validator = importlib.util.module_from_spec(validator_spec)
        validator_spec.loader.exec_module(validator)
        exec(compile(adapted, str(ROOT / 'summarize_performance.py'), 'exec'), validator.__dict__)
        validator.EXPECTED_COMMANDS = {}
        validator.EXPECTED_FORMAL_ENTRY = REMOTE_NATIVE + '/pretrain_gpt.py'
        seeds = self.contract['performance']['independent_seeds']
        require(perf['seeds'] == seeds and perf['orders'] == list(ORDERS)
                and len(perf['pairs']) == 3 and len(perf['completed_cases']) == 9,
                'complete declared seeds/rotation/process coverage required')
        ratios = {'vs_native': [], 'vs_control': [], 'control_vs_native': []}
        for pair, seed, order in zip(perf['pairs'], seeds, ORDERS):
            require(pair['seed'] == seed and pair['order'] == order, 'performance pair seed/order differs')
            means, commands, models = {}, {}, {}
            for arm in order:
                name = 'main-' + arm + '-perf-' + self.version + '-' + suffix + '-' + str(seed)
                folder = self.base / name
                command, entry = self.entry(folder, arm, 120, seed)
                activation, grad = self.hooks(folder, arm, 120, wrapper)
                expected = ['/root/miniconda3/bin/python', '-u', REMOTE + '/' + wrapper.name,
                            '--swiglu-receipt', REMOTE + '/' + name + '/swiglu.json',
                            '--gradnorm-launcher', REMOTE_GRAD + '/launch_gradnorm-04.py',
                            '--bf16-root', REMOTE_BF, '--gradnorm-receipt', REMOTE + '/' + name + '/gradnorm.json',
                            '--megatron-root', REMOTE_NATIVE, '--contract', REMOTE + '/contracts/main.json',
                            '--entry-evidence', REMOTE + '/' + name + '/entry.json']
                if arm != 'native':
                    expected += ['--candidate', REMOTE_BF + '/candidates/0008/candidate.py',
                                 '--gradnorm-candidate', REMOTE_GRAD + '/candidate.py']
                if arm == 'optimized':
                    expected += ['--swiglu-candidate', REMOTE + '/candidate-' + self.version + '.py']
                expected += self.contract['training_argv'] + ['--train-iters', '120', '--seed', str(seed),
                                                              '--log-interval', '10']
                require(command['command'] == expected and command['stage29_arm'] == arm
                        and command['wrapper_sha256'] == sha(wrapper), 'exact performance outer argv differs')
                validator.EXPECTED_COMMANDS[str(folder)] = expected
                frozen_result = validator.validate_case(
                    folder, command['mode'], seed, self.contract, sha(self.contract_path),
                    self.contract['candidate_sha256'])
                require(not any((folder / item).exists() for item in ('snapshots', 'profile', 'activation')),
                        'timed case contains diagnostic artifacts')
                intervals = parser.parse_intervals((folder / 'stdout.log').read_text(encoding='utf-8'))
                means[arm] = statistics.mean(row['mean_iteration_ms'] for row in intervals if row['retained'])
                close(frozen_result['mean_iteration_ms'], means[arm],
                      'frozen original-validator mean differs from independent raw parsing')
                recorded = pair['arms'][arm]
                close(recorded['mean_iteration_ms'], means[arm], 'performance raw-log mean differs')
                require(recorded['intervals'] == intervals and recorded['model'] == entry['models'][0]
                        and recorded['case'] == name and recorded['seed'] == seed
                        and recorded['hooks']['swiglu'] == activation and recorded['hooks']['gradnorm'] == grad,
                        'summary timing/model/hook records differ from raw evidence')
                for key, digest in recorded['raw_sha256'].items():
                    require(sha(folder / key) == digest, 'timed raw evidence bytes differ: ' + key)
                for key, digest in recorded['hooks']['raw_sha256'].items():
                    require(sha(folder / key) == digest, 'timed hook evidence bytes differ: ' + key)
                commands[arm], models[arm] = command, entry['models']
            require(models['native'] == models['control'] == models['optimized'], 'timed models differ')
            for first, second in zip(order, order[1:]):
                require(datetime.fromisoformat(commands[first]['finished_utc'])
                        <= datetime.fromisoformat(commands[second]['started_utc']),
                        'timed processes overlap or rotation order differs')
            values = {'vs_native': means['native'] / means['optimized'],
                      'vs_control': means['control'] / means['optimized'],
                      'control_vs_native': means['native'] / means['control']}
            for metric, value in values.items():
                close(pair[metric], value, 'per-seed ratio differs: ' + metric)
                ratios[metric].append(value)
        recomputed = {}
        for metric, values in ratios.items():
            ratio = math.exp(statistics.mean(math.log(value) for value in values))
            recomputed[metric] = {'ratio': ratio, 'gain_percent': 100 * (ratio - 1),
                                  'all_seeds_faster': all(value > 1 for value in values)}
            close(perf[metric]['ratio'], ratio, 'geomean ratio differs: ' + metric)
            close(perf[metric]['gain_percent'], recomputed[metric]['gain_percent'], 'gain differs: ' + metric)
            require(perf[metric]['all_seeds_faster'] is recomputed[metric]['all_seeds_faster'],
                    'all-seed positive-result flag differs: ' + metric)
        promoted = recomputed['vs_control']['ratio'] > 1.01 and recomputed['vs_control']['all_seeds_faster']
        require(perf['promoted'] is promoted and perf['decision'] == ('promoted' if promoted else 'not_promoted')
                and perf['promotion_rule'] == {'gain_vs_control_strictly_greater_than': 0.01,
                                               'all_three_seeds_faster_vs_control': True},
                'promotion contradicts recomputed strict >1% and all-seed gate')
        return {'available': True, 'timed_processes': 9, 'promoted': promoted, **recomputed,
                'path': str(path), 'sha256': sha(path)}


def verify(args):
    archive = Archive(args.base, args.version)
    archive.foundations()
    archive.backup()
    if args.probe:
        probe = Path(args.probe)
        if not probe.is_absolute():
            probe = archive.base / probe
    else:
        choices = [archive.base / ('probe-' + args.version + '-v3.json'),
                   archive.base / ('probe-' + args.version + '-v2.json'),
                   archive.base / ('probe-' + args.version + '.json')]
        probe = next((path for path in choices if path.is_file()), choices[0])
    require(probe.resolve().is_relative_to(archive.base), 'selected probe must belong to evidence archive')
    selected = archive.probe(probe)
    audit_path, audit_summary, comparisons = archive.audit(args.audit_suffix)
    performance = archive.performance(args.run_suffix, audit_path, audit_summary)
    return {'status': 'verified', 'version': args.version, 'probe': selected,
            'comparisons': comparisons, 'performance': performance,
            'compiled_extension': archive.extension, 'binary_bytes_checked': archive.binary_bytes_checked,
            'snapshot_files_hashed': archive.snapshot_present,
            'snapshot_files_not_present_locally': archive.snapshot_missing,
            'raw_backup': archive.backup_summary,
            'gpu_rerun': False, 'tensor_snapshot_values_recomputed': False,
            'scope': 'CPU archive revalidation of every published numerical check/tensor record, '
                     'all source/contract/controller provenance, and raw native timing logs. '
                     'No training/GPU rerun and no tensor snapshot arithmetic replay. '
                     'Available raw snapshots and extension binaries are checked by SHA256; '
                     'missing raw bytes remain explicitly reported.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--probe', help='archive filename; default preference is v3, v2, then original')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    parser.add_argument('--base', type=Path, default=DEFAULT_BASE)
    args = parser.parse_args()
    try:
        require(args.version.isdigit() and all(value.replace('-', '').isalnum()
                for value in (args.audit_suffix, args.run_suffix)), 'simple immutable version/suffix required')
        # Avoid writing __pycache__ while importing the pure standard-library parser.
        sys.dont_write_bytecode = True
        result = verify(args)
    except Exception as error:
        result = {'status': 'rejected', 'version': args.version,
                  'error_type': type(error).__name__, 'error': str(error), 'gpu_rerun': False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'verified' else 1


if __name__ == '__main__':
    raise SystemExit(main())
