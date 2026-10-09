"""CPU archive verification of fixed MoE numerical and timing receipts.

Never imports Torch, a candidate, or a GPU experiment controller. Available
original bytes are hashed; CI explicitly reports unavailable large raw files.
This checks archived GPU comparisons, not a replay of their tensor arithmetic.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import statistics

HERE = Path(__file__).resolve().parent
BF = HERE.parent
FORMAL = BF.parent / 'megatron_training_entry'
REMOTE = '/tmp/kda-moe-weighted-stage34/'
FROZEN = {
    'contract.json': '31a09fd357fbe386b493ea1efa348a870e81326c6cfe3e757bf9ae9b7478e263',
    'launch_weighted.py': '65f01f6aa474ae08c3351106df245477ac6c224d951408fc938aa7b19eba48ad',
    'run_case.py': 'a85cb3dd5a9f32981e0d20e068be31ebe46467266062e93284ff58d3f45fc797',
    'compare.py': '52481dce035a59c523cc137d968a998eeb4f20ff6f3b3590c3492fa5f7c50635',
    'probe.py': 'c5c45eca4ea3f56de0a1cfddf2844deb37c6689cbabe5c48e4ada743cb010387',
    'probe_offset.py': 'a672dc9c1015b38a42416d083dd4f16185459b39fe002a456d8b1e761f370840',
    'seal_audit.py': '3bcfc8abba6b51e0d269fb1103aa0e8cfeb0298374bc9df072d39719ce5bd0dd',
    'run_performance.py': '2775cbd70ce6e21d8b310a666609885f54893c8de53d4918098e9a61318bfd3b',
    'diagnose.py': '3a740d3a0bbce2a3b77a3b8a1df861e2bfed63d6eb0ab7c1ab837c09013c92d7',
}
GROUPS = {'initial': 56, 'actual_start_model_parameters': 56, 'actual_start_master_parameters': 56,
    'master_initialization': 112, 'model_initialization': 112, 'batches': 15,
    'output': 3, 'derived_loss': 3, 'wgrads': 168, 'params': 168, 'masters': 168,
    'master_model_consistency': 336, 'routing': 36}
ORDERS = [['native', 'control', 'optimized'], ['optimized', 'control', 'native'],
          ['control', 'native', 'optimized']]


def require(value, message):
    if not value:
        raise ValueError(message)


SHA_CACHE = {}

def sha(path):
    path=Path(path)
    stamp=path.stat()
    key=(stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns)
    if key in SHA_CACHE:
        return SHA_CACHE[key]
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    value=h.hexdigest()
    after=path.stat()
    require((after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==key,"original file changed during hashing")
    SHA_CACHE[key]=value
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(left, right):
    require(math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), 'statistics differ')


def local_original(value, base):
    if value.startswith(REMOTE):
        rel = PurePosixPath(value[len(REMOTE):])
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe raw path')
        for folder in (base, base / 'raw-backup', HERE):
            path = folder.joinpath(*rel.parts)
            if path.is_file():
                return path
        return None
    roots = {'/tmp/kda-bf16-continuation/megatron_bf16/': BF,
             '/tmp/kda-bf16-continuation/megatron_training_entry/': FORMAL,
             '/tmp/kda-gradnorm-stage26/': BF / 'evidence/gradnorm-stage26'}
    for remote, local in roots.items():
        if value.startswith(remote):
            rel = PurePosixPath(value[len(remote):])
            require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe dependency path')
            path = local.joinpath(*rel.parts)
            return path if path.is_file() else None
    raise ValueError('unexpected original input path: ' + value)


def verify(version, base, require_raw=False, confirmation=False):
    for name, digest in FROZEN.items():
        require(sha(HERE/name) == digest, 'frozen controller changed: ' + name)
    candidate = HERE/'candidates'/version/'candidate.py'
    generation = read(candidate.with_name('generation.json'))
    require(generation['candidate_sha256'] == sha(candidate), 'generated candidate differs')
    response = BF/'evidence'/('cc-moe-weighted-'+version+'.json')
    require(generation['response_sha256'] == sha(response) and read(response)['is_error'] is False,
            'original Claude Code generation missing')
    contract = read(HERE/'contract.json')
    require(contract['correctness']['atol'] == 3e-4 and contract['correctness']['rtol'] == 1e-3,
            'frozen tolerance changed')
    audit = read(base/('audit-'+version+'.json'))
    require(audit['passed'] is True and audit['candidate_sha256'] == sha(candidate)
            and audit['contract_sha256'] == sha(HERE/'contract.json'), 'sealed audit differs')
    for name, digest in audit['controller_sha256'].items():
        require(FROZEN[name] == digest == sha(HERE/name), 'audit controller changed')
    raw_rows = {}
    comparisons = {}
    for name in ('comparison-native-001.json', 'comparison-control-001.json'):
        negative = read(base/name)
        require(negative['passed'] is False and negative['check_summary'] ==
                {'total':2627,'passed':2625,'failed':2}, 'first candidate rejection missing')
        bad = [r for r in negative['tensor_comparisons'] if not r['passed']]
        require(len(bad) == 2 and sum(r['elements_outside_tolerance'] for r in bad) == 8
                and all(r['path'].startswith('step 3/wgrads/model_chunk0/embedding.') for r in bad),
                'original first-candidate failure differs')
        for row in negative['input_files'] + negative['route_input_files']:
            raw_rows[row['path']] = row
    for name, arms in [('comparison-admission01.json', ('native', 'control')),
                       ('comparison-native-'+version+'.json', ('native', 'optimized')),
                       ('comparison-control-'+version+'.json', ('control', 'optimized'))]:
        value = read(base/name)
        require(audit['comparisons'][name]['sha256'] == sha(base/name), 'comparison bytes differ')
        require(value['passed'] is True and (value['left_arm'], value['right_arm']) == arms
                and value['atol'] == 3e-4 and value['rtol'] == 1e-3
                and value['comparison_device'] == 'CPU float64 copies', 'original numerical gate differs')
        checks, tensors = value['checks'], value['tensor_comparisons']
        require(len(checks) == 2627 and all(r['passed'] is True for r in checks)
                and len(tensors) == 1289 and dict(Counter(r['group'] for r in tensors)) == GROUPS
                and len({r['path'] for r in tensors}) == len(tensors), 'numerical coverage differs')
        for row in tensors:
            require(row['shape_dtype_layout_match'] is row['all_elements_finite'] is row['passed'] is True
                    and row['elements_outside_tolerance'] == 0, 'failed numerical tensor')
            require(row['atol'] == (0 if row['exact_required'] else 3e-4)
                    and row['rtol'] == (0 if row['exact_required'] else 1e-3), 'per-tensor gate changed')
            if row['exact_equal']:
                require(row['native_sha256'] == row['optimized_sha256']
                        and row['elements_not_exactly_equal'] == row['max_absolute_error'] == 0,
                        'exact tensor receipt inconsistent')
        require(len(value['input_files']) == 35 and len(value['route_input_files']) == 48,
                'original comparison inputs missing')
        for row in value['input_files'] + value['route_input_files']:
            previous = raw_rows.setdefault(row['path'], row)
            require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                    'same original input differs across comparisons')
        comparisons[name] = {'checks': len(checks), 'tensors': len(tensors),
                            'all_exact': all(r['exact_equal'] for r in tensors)}
    probe = read(base/('probe-'+version+'.json'))
    offset = read(base/('offset-'+version+'.json'))
    diagnostic = read(base/('diagnostics-'+version+'.json'))
    require(probe['status'] == 'passed' and len(probe['checks']) == 44 and len(probe['guards']) == 2
            and all(r['passed'] for r in probe['checks']+probe['guards']), 'independent probe failed')
    require(offset['passed'] is True and len(offset['checks']) == 2
            and all(r['pointer_alignment_mod16'] == [2, 2] and r['passed'] for r in offset['checks']),
            'real unaligned views not verified')
    require(diagnostic['status'] == 'complete' and len(diagnostic['checks']) == 48
            and all(all(z['exact_equal'] for z in r['native_replay'].values())
                    and all(z['within_original_tolerance'] for z in r['checks'].values())
                    and r['fallback_calls'] == 0 for r in diagnostic['checks']), '48 real replay cases missing')
    require(diagnostic['candidate_sha256'] == sha(candidate)
            and diagnostic['verifier_sha256'] == sha(HERE/'diagnose.py'), 'diagnostic provenance differs')
    for name in ('native-admission01','control-admission01','optimized-audit-001',
                 'native-diagnostics001','optimized-audit-002','optimized-audit-'+version):
        weighted = read(base/name/'weighted.json')
        for rel, row in weighted['raw_files'].items():
            raw_rows[REMOTE+name+'/'+rel] = dict(row,path=REMOTE+name+'/'+rel)
    diagnostic_weighted = read(base/'native-diagnostics001/weighted.json')
    for index, row in enumerate(diagnostic['checks'],1):
        require(row['index'] == index and row['capture_sha256'] ==
                diagnostic_weighted['raw_files']['activation/capture-%03d.pt'%index]['sha256'],
                'actual native capture identity differs')
    for r, filename, controller in [(probe, 'probe-'+version+'.json', 'probe.py'),
                                     (offset, 'offset-'+version+'.json', 'probe_offset.py')]:
        require(r['candidate_sha256'] == sha(candidate) and r['verifier_sha256'] == sha(HERE/controller),
                'probe identity differs')
    require(audit['probe_sha256'] == sha(base/('probe-'+version+'.json'))
            and audit['offset_probe_sha256'] == sha(base/('offset-'+version+'.json')), 'probe bytes differ')
    timing_base=base/'confirmation' if confirmation else base
    performance = read(timing_base/('performance-'+version+'.json'))
    require(performance['status'] == 'validated' and performance['seeds'] == [34102,34103,34104]
            and len(performance['pairs']) == 3 and len(performance['completed_cases']) == 9,
            'complete rotated nine-process timing required')
    require(performance['candidate_sha256'] == sha(candidate)
            and performance['contract_sha256'] == sha(HERE/'contract.json')
            and performance['runner_sha256'] == sha(HERE/'run_performance.py')
            and performance['audit_sha256'] == sha(base/('audit-'+version+'.json')), 'performance provenance differs')
    parser_path = FORMAL/'summarize_performance.py'
    require(sha(parser_path) == '6c2d1f7955f7e474a3de0b9e4edc156bc9ba90b5f23c7a5fce696ee736dcd2d3', 'native log parser differs')
    spec = importlib.util.spec_from_file_location('moe_archive_native_parser', parser_path)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    source_manifest = read(FORMAL/'source_manifest.json')
    for seed, order, pair in zip(performance['seeds'], ORDERS, performance['pairs']):
        require(pair['seed'] == seed and pair['order'] == order, 'timing seed/order differs')
        for arm in order:
            case = pair['cases'][arm]; folder = timing_base/case['case']
            for name, digest in case['raw_sha256'].items():
                require(sha(folder/name) == digest, 'original timing input differs')
            entry, command, weighted, grad = (read(folder/name) for name in
                ('entry.json','command.json','weighted.json','gradnorm.json'))
            require(entry['native_sources_sha256'] == source_manifest['files']
                    and entry['megatron_commit'] == contract['megatron_commit'], '600 native source identities differ')
            require(command['steps'] == 120 and command['seed'] == seed and command['stage34_arm'] == arm
                    and command['audit'] is command['profile'] is False and command['exit_code'] == 0,
                    'timing process differs')
            require(weighted['calls'] == 1920 and weighted['sort_compat_calls'] == 480
                    and weighted['router_calls'] == 0 and weighted['expert_calls'] == weighted['configurations'] == []
                    and weighted['fallback_calls'] == 0 and weighted['enabled'] is (arm == 'optimized')
                    and weighted['candidate_calls'] == (1920 if arm == 'optimized' else 0), 'weighted dispatch differs')
            require(grad['calls'] == (0 if arm == 'native' else 120) and grad['norm_audit'] == [],
                    'gradnorm instrumentation in timing')
            for name, digest in case['dispatch_sha256'].items():
                require(sha(folder/name) == digest, 'original dispatch bytes differ')
            intervals = parser.parse_intervals((folder/'stdout.log').read_text(encoding='utf-8'))
            require(case['intervals'] == intervals, 'original intervals differ')
            close(case['mean_iteration_ms'], statistics.mean(r['mean_iteration_ms'] for r in intervals if r['retained']))
        for left, right in zip(order, order[1:]):
            require(datetime.fromisoformat(pair['cases'][left]['finished_utc']) <=
                    datetime.fromisoformat(pair['cases'][right]['started_utc']), 'timing processes overlap')
    for metric, left, right in [('vs_native','native','optimized'),('vs_control','control','optimized'),
                                ('control_vs_native','native','control')]:
        ratios = [r['cases'][left]['mean_iteration_ms']/r['cases'][right]['mean_iteration_ms'] for r in performance['pairs']]
        ratio = math.exp(statistics.mean(math.log(z) for z in ratios))
        close(performance[metric]['ratio'],ratio); close(performance[metric]['gain_percent'],100*(ratio-1))
        require(performance[metric]['all_seeds_faster'] is all(z>1 for z in ratios), 'all-seed promotion differs')
    promoted = performance['vs_control']['ratio'] > 1.01 and performance['vs_control']['all_seeds_faster']
    require(performance['promoted'] is promoted, 'fixed promotion rule differs')
    available, missing = [], []
    for value, row in raw_rows.items():
        local = local_original(value, base)
        if local is None:
            missing.append(value)
        else:
            require(local.stat().st_size == row['bytes'] and sha(local) == row['sha256'], 'original raw bytes differ')
            available.append(value)
    if require_raw:
        require(not missing, 'complete original raw bytes required locally')
    return {'status':'receipt_verified','version':version,'confirmation':confirmation,'comparisons':comparisons,
        'vs_control':performance['vs_control'],'vs_native':performance['vs_native'],'promoted':promoted,
        'available_original_inputs':len(available),'missing_original_inputs':missing,
        'original_byte_coverage_complete':not missing,'verifier_sha256':sha(__file__),
        'gpu_rerun':False,'arithmetic_replay':False,
        'scope':'Recorded fixed GPU gates and original logs verified on CPU; missing large raw files stay explicit in CI.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',required=True);p.add_argument('--base',type=Path,default=HERE/'evidence')
    p.add_argument('--require-raw',action='store_true');p.add_argument('--confirmation',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.version,a.base,a.require_raw,a.confirmation),indent=2))


if __name__=='__main__':
    main()
