"""Add sealed stage30/31 evidence to a skill clone without changing older pages.

Run after the root seals both decisions and CPU archive verification receipts.
This writes an additive evidence bundle, two source pages, two kernel pages,
and append-only SKILL/ledger guidance. It never builds or changes a Git patch.

Each decision must contain status, promoted, candidate_versions, scope and
rounds. A round identifies model_name, version, audit_suffix, run_suffix,
verification_path (relative to its fresh archive), verification_sha256.
If promoted is true, selected_candidate identifies a promoted fresh version.
All original raw bytes must be present and are hashed again before packaging.
Stage30 has wide/deep rounds; stage31 has main round(s). Invalid historical
metadata remains negative evidence and never contributes performance_claims.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics


BF = Path(__file__).resolve().parent
STAGE30 = BF / 'swiglu_scale'
STAGE31 = BF / 'attention'
BUNDLE = 'bi-v150-corex42-stage30-31'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
SOURCE_IDS = {30: 'exp-bi-v150-corex42-stage30', 31: 'exp-bi-v150-corex42-stage31'}
PAGE_NAMES = {30: 'swiglu-heldout-megatron-bi-v150', 31: 'dense-attention-megatron-bi-v150'}
EXCLUDED_SUFFIXES = {'.pt', '.pth', '.so', '.dll', '.pyd', '.a', '.o', '.obj', '.pyc',
                     '.gz', '.zip', '.tar', '.7z', '.bin', '.safetensors', '.cubin',
                     '.fatbin', '.bc', '.partial'}
EXCLUDED_DIRECTORIES = {'snapshots', 'raw-backup', '__pycache__', 'deps',
                        'objects', '.objects', '.object-store', '.git'}
RAW_HASH_CACHE = {}


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


def write_new(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(content)


def scalar(value):
    return json.dumps(str(value), ensure_ascii=False)


def safe_relative(value):
    path = PurePosixPath(str(value))
    require(not path.is_absolute() and '..' not in path.parts and bool(path.parts),
            'safe relative receipt path required')
    return path


def decision(path):
    value = read(path)
    require(all(key in value for key in ('status', 'promoted', 'candidate_versions', 'scope', 'rounds')),
            'sealed decision requires status/promoted/candidate_versions/scope/rounds')
    require(value['status'] not in ('running', 'initializing', 'pending', 'in_progress')
            and isinstance(value['rounds'], list) and value['rounds'], 'decision is not sealed')
    require(type(value['promoted']) is bool and isinstance(value['candidate_versions'], list)
            and value['candidate_versions'] and all(type(version) is str and version.isdigit()
            and len(version) == 3 for version in value['candidate_versions'])
            and len(value['candidate_versions']) == len(set(value['candidate_versions'])),
            'sealed boolean promotion and unique explicit three-digit candidate versions required')
    return value


def verify_raw_archive(stage, archive, model, version, suffix, audit, proof):
    """Hash actual original bytes now; a backup receipt alone cannot admit a round."""
    count = proof['snapshot_inventory_records']
    require(type(count) is int and count > 0 and proof['snapshot_files_missing'] == 0
            and proof['snapshot_files_available'] == count,
            'formal delivery requires every original snapshot present in the final CPU proof')
    key = model if stage == 30 else version
    inventory_path = archive / ('raw-inventory-' + key + '.json')
    receipt_path = archive / ('raw-backup-' + key + '-receipt.json')
    rows, receipt = read(inventory_path), read(receipt_path)
    require(isinstance(rows, list) and rows, 'nonempty original raw inventory required')
    remote = '/tmp/kda-swiglu-scale-stage30' if stage == 30 else '/tmp/kda-attention-stage31'
    folders = {model + '-' + arm + '-audit-' + version + '-' + suffix
               for arm in ('native', 'control', 'optimized')}
    indexed, unique = {}, {}
    for row in rows:
        rel = safe_relative(row['path'])
        require(len(rel.parts) >= 3
                and ((rel.parts[0] in folders and rel.parts[1] == 'snapshots')
                     or (stage == 31 and rel.parts[:2] == ('native-default-capture-02', 'attention')
                         and rel.suffix == '.pt'))
                and row['path'] not in indexed and type(row['bytes']) is int and row['bytes'] >= 0
                and len(row['sha256']) == 64 and all(char in '0123456789abcdef' for char in row['sha256']),
                'raw inventory path, size, SHA or coverage scope differs')
        indexed[row['path']] = row
        previous = unique.setdefault(row['sha256'], row)
        require(previous['bytes'] == row['bytes'], 'same raw SHA has different byte sizes')
        choices = [archive.joinpath(*rel.parts), (archive / 'raw-backup').joinpath(*rel.parts)]
        raw = next((path for path in choices if path.is_file()), None)
        require(raw is not None and raw.resolve().is_relative_to(archive),
                'actual original raw file is missing or outside the archive: ' + row['path'])
        stat = raw.stat()
        require(stat.st_size == row['bytes'], 'original raw byte size differs: ' + row['path'])
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity not in RAW_HASH_CACHE:
            RAW_HASH_CACHE[identity] = sha(raw)
        require(RAW_HASH_CACHE[identity] == row['sha256'], 'actual raw SHA256 differs: ' + row['path'])
        object_path = archive / 'raw-backup/objects' / row['sha256']
        require(object_path.is_file() and raw.samefile(object_path),
                'raw original path is not linked to its verified SHA256 object')
    require(receipt['status'] == 'verified' and receipt['audit_suffix'] == suffix
            and receipt['remote_root'] == remote and receipt['remote_deleted'] is False
            and receipt['inventory_sha256'] == sha(inventory_path) and receipt['files'] == rows
            and receipt['verified_files'] == len(rows)
            and receipt['raw_bytes'] == sum(row['bytes'] for row in rows)
            and receipt['unique_files'] == len(unique)
            and receipt['unique_bytes'] == sum(row['bytes'] for row in unique.values())
            and receipt['deduplication'] == 'shared SHA256 objects and hardlinks'
            and receipt['transfer_archive_retained'] is False
            and receipt['gpu_timing_finished_acknowledged'] is True
            and receipt['backup_script_sha256'] == sha((STAGE30 if stage == 30 else STAGE31) / 'backup_raw.py')
            and (receipt['model_name'] == model if stage == 30 else receipt['version'] == version),
            'full raw backup receipt identity differs')
    backup = proof['raw_backup']
    require(backup['inventory_available'] is True and backup['receipt_available'] is True
            and backup['inventory_sha256'] == sha(inventory_path)
            and backup['inventory_files'] == backup['files_available'] == len(rows)
            and backup['files_missing'] == 0, 'final CPU proof lacks full original raw byte coverage')
    required = {}
    labels = {'controlled': '', 'native_optimized': 'native-optimized-', 'native_control': 'native-control-'}
    for label, prefix in labels.items():
        item = audit['comparisons'][label]
        name = 'comparison-' + prefix + model + '-' + version + '-' + suffix + '.json'
        require(item['path'] == remote + '/' + name, 'numerical comparison path differs from this exact round')
        comparison_path = archive / name
        require(sha(comparison_path) == item['sha256'], 'original comparison receipt bytes changed')
        comparison = read(comparison_path)
        require(comparison['passed'] is True and comparison['check_summary']['failed'] == 0
                and all(row['passed'] is True for row in comparison['checks'])
                and all(row['passed'] is True for row in comparison['tensor_comparisons'])
                and all(row['passed'] is True for row in comparison['raw_value_comparisons']),
                'complete original raw numerical receipts failed')
        for row in comparison['input_files']:
            if row['path'].startswith(remote + '/') and '/snapshots/' in row['path']:
                rel = row['path'][len(remote) + 1:]
                previous = required.setdefault(rel, row)
                require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                        'original comparisons disagree about the same raw snapshot')
    if stage == 31:
        for name, row in audit['provenance']['native_capture_raw_files'].items():
            rel = 'native-default-capture-02/attention/' + name
            require(rel not in required, 'duplicate native capture identity')
            required[rel] = row
    require(len(required) == count, 'final CPU snapshot coverage differs from all original comparison/capture rows')
    for name, row in required.items():
        require(name in indexed and indexed[name]['bytes'] == row['bytes']
                and indexed[name]['sha256'] == row['sha256'],
                'raw byte inventory omits or changes a required original snapshot/capture')
    return {'original_files_hashed': len(rows), 'required_snapshot_files_hashed': len(required),
            'inventory_sha256': sha(inventory_path), 'backup_receipt_sha256': sha(receipt_path),
            'missing_original_files': 0}


def bind_decision(stage, sealed, rounds):
    identities = {(row['model_name'], row['version']) for row in rounds}
    require(len(identities) == len(rounds), 'duplicate fresh model/candidate decisions')
    versions = {row['version'] for row in rounds}
    declared = set(sealed['candidate_versions'])
    require(versions.issubset(declared), 'sealed candidate versions omit a fresh validated round')
    for version in declared:
        candidate = (BF / 'evidence/swiglu-stage29/candidate-004.py' if stage == 30 and version == '004'
                     else STAGE31 / ('candidates/' + version + '/candidate.py') if stage == 31 else None)
        require(candidate is not None and candidate.is_file(), 'declared candidate version has no actual source')
    require(sealed['promoted'] is any(row['promoted'] for row in rounds),
            'sealed promotion contradicts fresh same-round performance gates')
    if sealed['promoted']:
        selected = sealed.get('selected_candidate')
        require(type(selected) is str and selected in declared
                and selected in {row['version'] for row in rounds if row['promoted']},
                'promoted decision must select a promoted fresh candidate version')
    elif sealed.get('selected_candidate') is not None:
        raise ValueError('unpromoted sealed decision cannot select a promoted candidate')
    rejected = sealed.get('rejected_versions', {})
    require(isinstance(rejected, dict)
            and not {row['version'] for row in rounds if row['promoted']}.intersection(rejected),
            'sealed rejected versions contradict fresh promoted candidates')
    for declared_round, actual in zip(sealed['rounds'], rounds):
        for key in ('promoted', 'candidate_sha256', 'contract_sha256', 'performance_sha256'):
            if key in declared_round:
                require(declared_round[key] == actual[key], 'sealed round metadata contradicts actual ' + key)


def load_round(stage, archive, item):
    model, version = item['model_name'], str(item['version'])
    audit_suffix, run_suffix = item['audit_suffix'], item['run_suffix']
    require((stage == 30 and model in ('wide', 'deep') and version == '004')
            or (stage == 31 and model == 'main' and version.isdigit() and len(version) == 3),
            'round model/candidate scope differs')
    require(all(value.replace('-', '').isalnum() for value in (audit_suffix, run_suffix)),
            'simple audit/run suffix required')
    archive = archive.resolve()
    rel = safe_relative(item['verification_path'])
    verification_path = archive.joinpath(*rel.parts)
    require(verification_path.resolve().is_relative_to(archive), 'verification path escaped archive')
    require(sha(verification_path) == item['verification_sha256'], 'sealed CPU verification bytes differ')
    proof = read(verification_path)
    require(proof['status'] == 'verified' and proof['gpu_rerun'] is False
            and proof['tensor_arithmetic_replayed'] is False, 'passing CPU archive verification required')
    require(proof['verifier_sha256'] == sha((STAGE30 if stage == 30 else STAGE31) / 'verify_receipts.py'),
            'CPU proof was not produced by the archived current verifier bytes')
    require((stage == 30 and proof['model_name'] == model)
            or (stage == 31 and proof['version'] == version), 'CPU verification round identity differs')
    audited = archive / ('audit-summary-' + model + '-' + version + '-' + audit_suffix + '.json')
    perf_path = archive / ('performance-' + model + '-' + version + '-' + run_suffix + '.json')
    audit, perf = read(audited), read(perf_path)
    require(perf['status'] == 'validated' and audit['passed'] is True
            and perf['model_name'] == audit['model_name'] == model
            and audit['suffix'] == audit_suffix and perf['run_suffix'] == run_suffix
            and perf['audit_sha256'] == proof['audit_sha256'] == sha(audited)
            and perf['audit_path'] == ('/tmp/kda-swiglu-scale-stage30/' if stage == 30
                                       else '/tmp/kda-attention-stage31/') + audited.name,
            'validated performance must bind the same-model audit-summary bytes')
    checked = proof['performance']
    require(checked['available'] is True and checked['sha256'] == sha(perf_path)
            and checked['timed_processes'] == 9 and checked['run_steps'] == 1080
            and checked['retained_steps'] == 900
            and checked.get('diagnostic_only', False) is False
            and checked.get('audit_metadata_binding_passed', True) is True,
            'diagnostic or invalid-metadata timing cannot enter published claims')
    require(proof['candidate_sha256'] == perf['candidate_sha256']
            and proof['contract_sha256'] == perf['contract_sha256'], 'verified source/contract identity differs')
    candidate = (BF / 'evidence/swiglu-stage29/candidate-004.py' if stage == 30
                 else STAGE31 / ('candidates/' + version + '/candidate.py'))
    require(sha(candidate) == perf['candidate_sha256'], 'actual selected candidate bytes differ from the fresh round')
    comparisons = proof['comparisons']
    require(set(comparisons) == {'controlled', 'native_optimized', 'native_control'},
            'complete three-way numerical receipt verification required')
    for name, summary in comparisons.items():
        original = audit['comparisons'][name]
        require(summary['checks'] == original['check_summary']['total']
                == original['check_summary']['passed']
                and original['check_summary']['failed'] == 0
                and summary['tensors'] == original['tensor_comparisons'],
                'published numerical counts contradict CPU-rechecked raw receipts')
    rows = {}
    for record in perf['completed_cases']:
        seed = int(record['seed'])
        arms = rows.setdefault(seed, {})
        require(record['arm'] in ('native', 'control', 'optimized') and record['arm'] not in arms,
                'duplicate/missing seed arm')
        time = record['mean_iteration_ms']
        require(math.isfinite(time) and time > 0, 'finite positive native interval mean required')
        arms[record['arm']] = time
    require(len(rows) == 3 and set(rows) == set(perf['seeds'])
            and all(set(arms) == {'native', 'control', 'optimized'} for arms in rows.values()),
            'complete three-seed/nine-process performance required')
    ratios = {}
    for metric, left, right in (('vs_native', 'native', 'optimized'),
                                ('vs_control', 'control', 'optimized'),
                                ('control_vs_native', 'native', 'control')):
        values = [arms[left] / arms[right] for arms in rows.values()]
        ratio = math.exp(statistics.mean(math.log(value) for value in values))
        result = {'ratio': ratio, 'gain_percent': 100 * (ratio - 1),
                  'all_seeds_faster': all(value > 1 for value in values)}
        for source in (perf[metric], checked[metric]):
            require(math.isclose(source['ratio'], ratio, rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(source['gain_percent'], result['gain_percent'], rel_tol=1e-12, abs_tol=1e-12)
                    and source['all_seeds_faster'] is result['all_seeds_faster'],
                    'geometric throughput result differs from all three raw means')
        ratios[metric] = result
    promoted = ratios['vs_control']['ratio'] > 1.01 and ratios['vs_control']['all_seeds_faster']
    require(perf['promoted'] is checked['promoted'] is promoted
            and perf['decision'] == ('promoted_tested_model_only' if promoted else 'not_promoted'),
            'promotion differs from strict >1% plus all-seeds same-round control rule')
    require(perf['promotion_rule'] == {'gain_vs_control_strictly_greater_than': 0.01,
                                       'all_three_seeds_faster_vs_control': True},
            'recorded promotion rule differs from the fixed admission gate')
    contract_path = archive / ('contracts/' + ('main' if stage == 31 else model) + '.json')
    contract = read(contract_path)
    require(sha(contract_path) == perf['contract_sha256'] and contract['model'] == perf['model'],
            'archived native model contract differs')
    fusion = '--no-masked-softmax-fusion' not in contract['training_argv']
    require(fusion is (stage == 31) and '--swiglu' in contract['training_argv']
            and '--no-bias-swiglu-fusion' not in contract['training_argv'],
            'stage30 inherited softmax False / stage31 native default True scopes differ')
    require(contract['correctness']['atol'] == 3e-4 and contract['correctness']['rtol'] == 1e-3
            and contract['correctness']['steps'] == 3, 'original numerical gate changed')
    raw_verification = verify_raw_archive(stage, archive, model, version, audit_suffix, audit, proof)
    return {'stage': stage, 'model_name': model, 'version': version,
            'candidate_sha256': perf['candidate_sha256'], 'contract_sha256': perf['contract_sha256'],
            'performance_path': perf_path, 'performance_sha256': sha(perf_path),
            'verification_path': verification_path, 'verification_sha256': sha(verification_path),
            'by_seed': rows, 'ratios': ratios, 'promoted': promoted, 'decision': perf['decision'],
            'comparisons': comparisons, 'model': perf['model'], 'scope': perf['scope'],
            'masked_softmax_fusion': fusion,
            'raw_verification': raw_verification,
            'snapshot_files_available': proof['snapshot_files_available'],
            'snapshot_files_missing': proof['snapshot_files_missing']}


def collect_files(decisions):
    selected, omitted = set(), []
    for root in (STAGE30, STAGE31):
        for folder, directories, names in os.walk(root):
            blocked = [name for name in directories
                       if name in EXCLUDED_DIRECTORIES or name.startswith('cpp-build-')]
            for name in blocked:
                omitted.append({'path': (Path(folder) / name).relative_to(BF).as_posix(),
                                'reason': 'raw/cache/build directory excluded'})
            directories[:] = [name for name in directories if name not in blocked]
            selected.update(Path(folder) / name for name in names)
    selected.update(decisions)
    selected.add(Path(__file__).resolve())
    selected.update((BF / 'prompts').glob('cc-attention-*.txt'))
    selected.update((BF / 'evidence').glob('cc-attention-*.json'))
    selected.update((BF / 'prompts').glob('cc-swiglu-004*.txt'))
    selected.update((BF / 'evidence').glob('cc-swiglu-004*.json'))
    selected.update(BF / name for name in
                    ('launch_training.py', 'kernel_adapter.py', 'run_case.py', 'compare_training.py',
                     'summarize_performance.py', 'invoke_cc.ps1',
                     'candidates/0008/candidate.py', 'gradnorm_candidates/001/candidate.py',
                     'evidence/gradnorm-stage26/launch_gradnorm-04.py',
                     'evidence/swiglu-stage29/candidate-004.py',
                     'evidence/swiglu-stage29/launch_swiglu-03.py',
                     'evidence/swiglu-stage29/run_candidate_audit.py',
                     'evidence/swiglu-stage29/run_performance.py',
                     'evidence/swiglu-stage29/probe-004-v3.json',
                     'evidence/swiglu-stage29/decision.json',
                     'evidence/swiglu-stage29/contracts/main.json',
                     'evidence/swiglu-stage29/contracts/wide.json',
                     'evidence/swiglu-stage29/contracts/deep.json',
                     'swiglu/candidates/004/candidate.py', 'swiglu/candidates/004/generation.json'))
    files = []
    for path in sorted(selected):
        require(path.is_file() and not path.is_symlink(), 'missing or linked experiment file: ' + str(path))
        rel = path.relative_to(BF)
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            omitted.append({'path': rel.as_posix(), 'bytes': path.stat().st_size,
                            'reason': 'raw tensor/binary/transfer archive excluded'})
        else:
            require(path.stat().st_size < 16 * 1024 * 1024, 'unexpected large small-evidence file')
            files.append((path, 'megatron_bf16/' + rel.as_posix()))
    for name in ('runtime_compat.py', 'source_manifest.json', 'summarize_performance.py'):
        path = BF.parent / 'megatron_training_entry' / name
        require(path.is_file(), 'missing original source/runtime/parser dependency')
        files.append((path, 'megatron_training_entry/' + name))
    return files, omitted


def timing_text(rounds):
    sections = []
    for row in rounds:
        model = row['model']
        text = (f"### {row['model_name']}, candidate {row['version']}\n\n"
                f"{model['num_layers']} layers, H{model['hidden_size']}, FFN{model['ffn_hidden_size']}, "
                f"S{model['sequence_length']}, V{model['vocab_size']}, batch{model['global_batch_size']}; "
                f"{model['parameters']:,} parameters.\n\n"
                '| Seed | Native ms/step | Existing control ms/step | Control + candidate ms/step |\n'
                '|---|---:|---:|---:|\n')
        for seed, values in sorted(row['by_seed'].items()):
            text += f"|{seed}|{values['native']:.2f}|{values['control']:.2f}|{values['optimized']:.2f}|\n"
        text += '\n'
        for metric, label in (('vs_control', 'Added candidate versus same-round control'),
                              ('vs_native', 'Combined optimizations versus same-round native'),
                              ('control_vs_native', 'Existing control versus same-round native')):
            ratio = row['ratios'][metric]
            text += (f"- {label}: {ratio['ratio']:.8f}x ({ratio['gain_percent']:+.2f}%); "
                     f"every seed faster: `{str(ratio['all_seeds_faster']).lower()}`.\n")
        text += f"\nRound decision: `{row['decision']}`. Promotion applies only to this tested model.\n\n"
        for name, result in row['comparisons'].items():
            text += (f"- {name}: {result['checks']} gates; {result['tensors']} tensor comparisons; "
                     f"all exact: `{result['all_tensors_exact_equal']}`; "
                     f"all tensor bytes equal: `{result['all_tensor_bytes_equal']}`.\n")
        text += (f"- Locally available/missing original numerical snapshots: "
                 f"{row['snapshot_files_available']}/{row['snapshot_files_missing']}. "
                 'CPU archive verification hashes available bytes and rechecks receipts; '
                 'it does not replay tensor arithmetic.\n')
        sections.append(text)
    return '\n'.join(sections)


def negative_text(stage):
    if stage == 30:
        return ('The original `complete` wide/deep v1 reports have an invalid audit metadata '
                'binding: the numerical-gate loop overwrote `path`, so performance `audit_path` '
                'and its SHA identify the final raw snapshot instead of the audit summary. '
                'Those original files and five frozen controller/contract files are preserved '
                'verbatim. Their recorded timing and even any original `promoted` flag are '
                '**diagnostic only**; they provide no validated performance claim. The repair '
                'renames exactly three variable references, preserves every numerical/source/'
                'model/precision/timer/promotion gate, and requires fresh three-way audits and '
                'nine-process performance for each model. Read `controller-repair.json`.\n')
    probe = read(STAGE31 / 'evidence/probe-full-001.json')
    require(probe['status'] == 'rejected', 'candidate001 negative receipt changed')
    sdpa = read(STAGE31 / 'evidence/sdpa-backends-01.json')
    lines = ['Candidate001 is independently rejected; retain its original code, prompt, '
             'response and probe, without inferring training throughput.']
    for row in sdpa['backends']:
        if 'checks' in row:
            output = row['checks']['output']
            lines.append(f"- Requested SDPA `{row['backend']}`: `{row['status']}`; "
                         f"output outside fixed tolerance: {output['bad_elements']}/{output['elements']}; "
                         f"maximum absolute output difference: {output['max_abs']}. "
                         'The API label does not independently establish an internal GPU kernel timeline.')
        else:
            lines.append(f"- Requested SDPA `{row['backend']}`: `{row['status']}`.")
    lines.append('Candidate002 passed standalone/full-update numerical receipts, but its original '
                 'complete-round record did not satisfy every-seed control improvement. The same '
                 'controller audit-path shadowing also invalidated its formal metadata binding. '
                 'It remains unpromoted historical negative evidence; its summary percentages '
                 'are not published performance claims. Candidate003 uses the corrected controller '
                 'and a fresh audit/performance archive. Native causal fused softmax remains the '
                 'reference; do not replace it with eager unfused Torch to claim an improvement.')
    return '\n\n'.join(lines) + '\n'


def source_front(stage, skill, bundle, date):
    title = ('BI-V150 frozen SwiGLU heldout validation' if stage == 30
             else 'BI-V150 native default fused full Attention iteration')
    text = (f'---\nid: {SOURCE_IDS[stage]}\ntitle: {scalar(title)}\n'
            'source_category: local-experiment\narchitectures: [bi-v150]\n'
            'tags: [kernel-fusion, jit-compilation]\n'
            f'recorded_at: {date}\nenvironment:\n  gpu: "Iluvatar BI-V150, device 0"\n'
            '  corex: "4.2.0"\n  torch: "2.4.1"\n  triton: "vendor 2.1.0"\n'
            f'  megatron: "{COMMIT}"\n'
            '  controller: "Independent frozen controllers; Claude Code CLI / Paratera Claude-Opus-4.8 candidates"\n'
            'evidence_files:\n')
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            text += f'  - path: {path.relative_to(skill).as_posix()}\n    sha256: {sha(path)}\n'
    return text + '---\n\n'


def kernel_front(stage, rounds):
    name = PAGE_NAMES[stage]
    text = (f'---\nid: kernel-{name}\ntitle: {scalar(name.replace("-", " "))}\n'
            'type: kernel\narchitectures: [bi-v150]\n'
            'tags: [kernel-fusion, jit-compilation]\n'
            + ('kernel_types: [elementwise, fused-kernel]\nlanguages: [triton, python]\n' if stage == 30
               else 'kernel_types: [attention, fused-kernel]\nlanguages: [python, cuda-cpp]\n')
            + 'confidence: experimental\nreproducibility: runnable\n'
            + ('related: [kernel-swiglu-megatron-bi-v150, lang-triton-iluvatar]\n' if stage == 30
               else 'related: [kernel-flash-attention-4, kernel-swiglu-megatron-bi-v150]\n')
            + f'sources: [{SOURCE_IDS[stage]}]\nperformance_claims:\n')
    for row in rounds:
        model = row['model']
        shape = (f"{model['num_layers']}L H{model['hidden_size']} FFN{model['ffn_hidden_size']} "
                 f"S{model['sequence_length']} V{model['vocab_size']} batch{model['global_batch_size']}; "
                 f"{model['parameters']} parameters; masked_softmax_fusion={row['masked_softmax_fusion']}")
        path = row['performance_path'].relative_to(BF).as_posix()
        for metric, label in (('vs_control', 'added candidate versus same-round existing control'),
                              ('vs_native', 'combined optimizations versus same-round native')):
            text += ('  - gpu: "Iluvatar BI-V150"\n    dtype: bf16\n'
                     f'    shape: {scalar(shape)}\n'
                     f"    metric: {scalar('Native synchronized formal wall throughput ratio; ' + label + '; ' + row['decision'])}\n"
                     f"    value: {row['ratios'][metric]['ratio']!r}\n"
                     f'    source_id: {SOURCE_IDS[stage]}\n'
                     f'    source_locator: {scalar("evidence/" + BUNDLE + "/megatron_bf16/" + path + "#" + metric)}\n')
    return text + '---\n\n'


def body(stage, sealed, rounds):
    scope = ('Stage30 retains `--no-masked-softmax-fusion` from its original fixed wide/deep '
             'contracts: native Attention softmax fusion is **False**. Its baseline activation '
             'is still the default compiled native SwiGLU, with no activation fusion weakened. '
             'Control is CE0008 + gradnorm001; only the frozen SwiGLU004 activation is added '
             'in the optimized arm. There is no shape-specific retuning.\n' if stage == 30 else
             'Stage31 enables proved original default native causal Softmax fusion **True** '
             'in every arm. S128/256/512 backend probes execute the vendor CUDA-extension '
             'forward/backward APIs. Control is CE0008 + gradnorm001 + frozen SwiGLU004; '
             'only the full Attention candidate differs in the optimized arm. It preserves '
             'vendor GEMM and fused-softmax operations; native RMSNorm remains unchanged.\n')
    text = ('# ' + ('Frozen SwiGLU on heldout models' if stage == 30 else 'Native fused Dense Attention on BI-V150')
            + '\n\n' + scope + '\n'
            + '**Stage30 and stage31 percentages are separate same-round comparisons. Do not '
              'add, multiply or splice them across the different model/backend contracts.** '
              'SwiGLU models are also distinct from older GELU13.7M trials.\n\n'
            + f"Sealed status: `{sealed['status']}`. Sealed decision scope: {sealed['scope']}\n\n"
            + '## Fresh validated rounds\n\n' + timing_text(rounds)
            + '\n## Rejected and historical evidence\n\n' + negative_text(stage)
            + '\n## Common measurement and migration limits\n\n'
            + 'Every validated performance round has three independent seeds and three rotated '
              'fresh-process arms. Each runs120 steps; first20 are excluded and ten original '
              'interval10 means are retained. The primary metric is the pinned training loop\'s '
              'GPU-synchronized `time.time` interval timer. Timing excludes audits, snapshots, '
              'profiling, evaluation and checkpoints. API `torch.cuda.Event` samples may include '
              'host/autograd submission gaps and are not isolated GPU kernel latency. Three seeds '
              'do not establish confidence intervals. Promotion requires all fixed update gates, '
              'strict >1% added throughput over the same-round control, and all three seeds faster.\n\n'
            + 'The source/model/data/DDP/SGD/native FP32 master path is pinned. BF16 model weights '
              'and intermediates, FP32 gradient accumulation and master weights, no autocast, '
              'TP/PP/CP1 and single BI-V150 define this scope. CPU float64 comparison tolerances '
              'remain atol3e-4/rtol1e-3, with strict initialization/input/master consistency checks. '
              'No production, convergence, AdamW, multi-GPU or unrestricted replacement claim follows.\n\n'
            + 'Portable NVIDIA ideas include tiling, dataflow fusion, staged rounding analysis and '
              'reducing host dispatch. TMEM/TMA/tcgen05/WGMMA/NVFP4 and H100/B200 performance '
              'numbers are not BI-V150 facts. Keep fresh input/output pointers and current streams; '
              'only immutable code/config metadata may be cached.\n')
    if stage == 31:
        text += ('\n## Exact native semantics\n\n'
                 'The real packed Q/K/V views are BF16[S128,B2,H8,D64], strides '
                 '[3072,1536,192,1], storage offsets0/64/128. Native `baddbmm` applies '
                 'scale0.125 and emits BF16 scores; the same vendor causal fused Softmax '
                 'emits BF16 probabilities, and native `bmm` emits BF16 context before '
                 'the output transpose/contiguous conversion. The default causal fused '
                 'branch ignores explicit mask contents and applies an internal upper '
                 'triangle. The older unfused fallback\'s BF16 -9984 sentinel and arbitrary '
                 'fully-masked row behavior are not automatically this fused contract. '
                 'SDPA/Flash internal math changes need independent proof, even when each '
                 'algorithm is mathematically attention.\n\n'
                 'Native source bytes for `dot_product_attention.py`, `fused_softmax.py` '
                 'and `utils.py`, including copyright notices, remain unchanged in the '
                 'bundle and are bound to the600-file source manifest. Backend/API route '
                 'receipts are not a GPU timeline.\n')
    text += f'\nSmall evidence: `evidence/{BUNDLE}`. Original raw tensors/binaries remain excluded; inventories and verified backup receipts are included.\n'
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill', type=Path, required=True)
    parser.add_argument('--stage30-decision', type=Path, default=STAGE30 / 'evidence/decision.json')
    parser.add_argument('--stage31-decision', type=Path, default=STAGE31 / 'evidence/decision.json')
    parser.add_argument('--stage30-archive', type=Path, default=STAGE30 / 'evidence/round-v2')
    parser.add_argument('--stage31-archive', type=Path, default=STAGE31 / 'evidence/round003')
    args = parser.parse_args()
    skill = args.skill.resolve()
    require((skill / 'SKILL.md').is_file() and (skill / 'references/iluvatar-migration-ledger.md').is_file(),
            'existing skill clone with original pages required')
    decisions = {30: decision(args.stage30_decision), 31: decision(args.stage31_decision)}
    rounds = {30: [load_round(30, args.stage30_archive, row) for row in decisions[30]['rounds']],
              31: [load_round(31, args.stage31_archive, row) for row in decisions[31]['rounds']]}
    for stage in (30, 31):
        bind_decision(stage, decisions[stage], rounds[stage])
    require({row['model_name'] for row in rounds[30]} == {'wide', 'deep'} and len(rounds[30]) == 2,
            'stage30 seals exactly one fresh wide and one fresh deep round')
    bundle = skill / 'evidence' / BUNDLE
    source_paths = {stage: skill / f'sources/experiments/bi-v150-corex42-stage{stage}.md' for stage in (30, 31)}
    kernel_paths = {stage: skill / ('wiki/kernels/' + PAGE_NAMES[stage] + '.md') for stage in (30, 31)}
    require(not any(path.exists() for path in (bundle, *source_paths.values(), *kernel_paths.values())),
            'immutable additive delivery already exists')
    files, omitted = collect_files((args.stage30_decision.resolve(), args.stage31_decision.resolve()))
    # Render and validate all decision-dependent text before the first mutation.
    bodies = {stage: body(stage, decisions[stage], rounds[stage]) for stage in (30, 31)}
    kernel_fronts = {stage: kernel_front(stage, rounds[stage]) for stage in (30, 31)}
    append_paths = (skill / 'SKILL.md', skill / 'references/iluvatar-migration-ledger.md')
    originals = {path: sha(path) for path in skill.rglob('*') if path.is_file() and '.git' not in path.parts}
    append_originals = {path: path.read_bytes() for path in append_paths}
    bundle.mkdir(exist_ok=False)
    write_new(bundle / '.gitattributes', '* -text -eol -whitespace\n')
    records = []
    for original, rel in files:
        destination = bundle / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        require(sha(destination) == sha(original), 'copied evidence bytes changed')
        records.append({'path': rel, 'bytes': destination.stat().st_size, 'sha256': sha(destination)})
    provenance = {'schema': 'bi-v150-stage30-31-additive-evidence-provenance-v1',
                  'source_ids': list(SOURCE_IDS.values()), 'asset_mode': 'verbatim',
                  'license': 'Source notices retained; native files inherit their upstream license',
                  'megatron_commit': COMMIT, 'packager_sha256': sha(__file__),
                  'decisions': {str(stage): {'path': str(path.relative_to(BF)), 'sha256': sha(path)}
                                for stage, path in ((30, args.stage30_decision.resolve()),
                                                    (31, args.stage31_decision.resolve()))},
                  'files': records, 'excluded_raw_files': omitted,
                  'raw_policy': 'Raw tensor/checkpoint/binary files, caches and transfer archives excluded. '
                                'Raw inventories and hash backup receipts are included.',
                  'legacy_policy': 'Invalid audit-path metadata and rejected numerical candidates remain '
                                   'verbatim negative evidence; they contribute no performance_claims.',
                  'fresh_raw_byte_verification': {str(stage): [row['raw_verification'] for row in rounds[stage]]
                                                  for stage in (30, 31)},
                  'existing_skill_files_sha256': {str(path.relative_to(skill)): digest
                                                 for path, digest in originals.items()},
                  'base_patch_modified': False}
    write_new(bundle / 'manifest.json', json.dumps(provenance, indent=2, ensure_ascii=False) + '\n')
    write_new(bundle / 'PROVENANCE.yaml', 'asset_mode: verbatim\nlicense: inherits-from-source-notices\n'
              f'upstream_sha: {COMMIT}\ngenerated_by: package_extension_evidence.py\n'
              'source_ids: [' + ', '.join(SOURCE_IDS.values()) + ']\nfiles:\n'
              + ''.join(f"  - local_path: {row['path']}\n    sha256: {row['sha256']}\n    mode: verbatim\n"
                        for row in records))
    for stage in (30, 31):
        date = decisions[stage].get('date', '2026-10-04')
        write_new(source_paths[stage], source_front(stage, skill, bundle, date) + bodies[stage])
        snippet = ('```python\nimport torch\n'
                   'from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl\n'
                   '# Keep the original compiled native SwiGLU activation for unpromoted shapes.\n'
                   'x = torch.randn(256, 2, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
                   'y = bias_swiglu_impl(x, None, False, False)\n'
                   'dx = torch.autograd.grad(y, x, torch.randn_like(y))[0]\n'
                   'assert y.shape == (256, 2, 2048) and dx.shape == x.shape\n```\n' if stage == 30 else
                   '```python\nimport torch\n# Exact default vendor causal fused-softmax API.\n'
                   'from megatron.core.fusions.fused_softmax import ScaledUpperTriangMaskedSoftmax\n'
                   '# Formal scores are native BF16 baddbmm results [B*H,S,S].\n'
                   'scores = torch.randn(16, 128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
                   'probs = ScaledUpperTriangMaskedSoftmax.apply(scores, 1.0)\n'
                   'dscores = torch.autograd.grad(probs, scores, torch.randn_like(probs))[0]\n'
                   'assert probs.shape == scores.shape and dscores.dtype == scores.dtype\n```\n')
        write_new(kernel_paths[stage], kernel_fronts[stage] + bodies[stage]
                  + '\n## Pinned native reference snippet\n\n' + snippet)
    guidance = ('\nFor BI-V150 heldout SwiGLU and native fused Attention, read '
                '[stage30](sources/experiments/bi-v150-corex42-stage30.md) and '
                '[stage31](sources/experiments/bi-v150-corex42-stage31.md). Stage30 retains '
                'its inherited softmax fusionFalse contract; stage31 proves and retains '
                'native default fusionTrue. Compare only fresh same-round controls and '
                'reject broken audit metadata even when timing appears faster.\n')
    with append_paths[0].open('ab') as stream:
        stream.write(guidance.encode())
    guidance = ('\n### Stages30/31: heldout models and native fused Attention\n\n'
                'Keep each model/backend contract distinct: stage30 fixed SwiGLU004 heldout '
                'validation has inherited softmax fusionFalse, stage31 full Attention uses '
                'proved native default fusionTrue. Preserve BF16 stage boundaries, native '
                'GEMMs/Softmax, all gradients and FP32 masters. SDPA availability does not '
                'establish numerical compatibility; broken audit metadata invalidates '
                'formal throughput claims. [Stage30](../sources/experiments/bi-v150-corex42-stage30.md) '
                'and [stage31](../sources/experiments/bi-v150-corex42-stage31.md).\n')
    with append_paths[1].open('ab') as stream:
        stream.write(guidance.encode())
    for path, digest in originals.items():
        if path in append_originals:
            require(path.read_bytes().startswith(append_originals[path]), 'existing appended bytes changed')
        else:
            require(sha(path) == digest, 'existing skill/NVIDIA/experiment page changed: ' + str(path))
    print(json.dumps({'status': 'packaged', 'bundle': BUNDLE, 'files': len(records),
                      'bundle_bytes': sum(row['bytes'] for row in records),
                      'new_source_pages': 2, 'new_kernel_pages': 2,
                      'existing_nonappend_files_byte_identical': True,
                      'base_patch_modified': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
