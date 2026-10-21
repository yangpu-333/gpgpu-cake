"""CPU-only review of the personal real Jiuge 7B CPT tracing contract.

Does not load weights, execute model code, start a profiler, or prove GPU math.
Independent model/bridge/full-step receipts are required. File SHA verification
establishes receipt/trace integrity, not truth of the producer's declarations.
Preparation and delivery are separate gates; the old 120-step performance gate
is not a tracing prerequisite. Delivery additionally requires all-rank evidence
from at least two distinct stable capture windows.
"""
import argparse
import hashlib
import json
from pathlib import Path
import posixpath
import re

from analyze_trace import analyze_rank_trace, load_trace

COMMIT = '5be9626709af2722333bf54797c954c09edeada3'
CONFIG_SHA = '3531fc5432a1f61e76e419ef773da7651f58d30a9f895d62bba3071b949e1d8a'
DATA_SHA = '6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903'
PREPARED_SHA = '38b6534e0a8ac5278c72adc69ce7f80f10c9bfc0dacc204bde8e73686efcd335'
FULL_SCOPE = {'data', 'forward', 'backward', 'gradient_communication', 'optimizer', 'waits'}


def positive_integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def check_contract(data, base_directory, max_artifact_bytes=1024 * 1024 * 1024):
    """Read small proof receipts and hash bounded local trace files, never weights."""
    base_directory = Path(base_directory).resolve()
    prerequisites, delivery = [], []
    checked, consumed = {}, 0

    def section(name):
        value = data.get(name)
        if not isinstance(value, dict):
            prerequisites.append(name + ' is missing or is not an object')
            return {}
        return value

    def require(values, keys, prefix, target=prerequisites):
        for key in keys:
            value = values.get(key)
            if value is None or value == '' or value == []:
                target.append(prefix + '.' + key + ' is not fixed')
            elif key.endswith('sha256') and not valid_sha(value):
                target.append(prefix + '.' + key + ' is not a SHA256 digest')

    def require_true(values, keys, prefix, target=prerequisites):
        for key in keys:
            if values.get(key) is not True:
                target.append(prefix + '.' + key + ' is unverified')

    def artifact(value, prefix, target, read_json=True):
        nonlocal consumed
        if not isinstance(value, dict):
            target.append(prefix + ' artifact reference is missing')
            return None
        require(value, ['path', 'sha256'], prefix, target)
        if not isinstance(value.get('path'), str) or not value['path'] or not valid_sha(value.get('sha256')):
            return None
        path = (base_directory / value['path']).resolve()
        try:
            if not path.is_file():
                raise ValueError('local artifact is unavailable: ' + str(path))
            if path not in checked:
                size = path.stat().st_size
                if size <= 0 or consumed + size > max_artifact_bytes:
                    raise ValueError('empty artifact or total artifact byte budget exceeded')
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    while block := stream.read(1024 * 1024):
                        digest.update(block)
                checked[path] = {'sha256': digest.hexdigest(), 'bytes': size}
                consumed += size
            if checked[path]['sha256'] != value['sha256']:
                raise ValueError('artifact SHA256 mismatch: ' + str(path))
            if read_json:
                if checked[path]['bytes'] > 4 * 1024 * 1024:
                    raise ValueError('proof JSON exceeds 4 MiB')
                result = json.loads(path.read_text(encoding='utf-8'))
                if not isinstance(result, dict):
                    raise ValueError('proof JSON is not an object')
                return result
            return path
        except (OSError, ValueError) as error:
            target.append(prefix + ': ' + str(error))
            return None

    if not isinstance(data, dict):
        raise ValueError('contract must be an object')
    if data.get('schema') != 'jiuge-7b-cpt-tracing-v1':
        prerequisites.append('unsupported tracing contract schema')
    scope = section('scope')
    if scope.get('personal_model') != 'Jiuge 7B' or scope.get('personal_task') != 'continued_pretraining':
        prerequisites.append('personal scope must be real Jiuge 7B continued pretraining')
    if scope.get('at_least_two_gpus_required') is not True:
        prerequisites.append('the at-least-two-GPU requirement cannot be relaxed')
    framework = section('framework')
    if framework.get('commit') != COMMIT:
        prerequisites.append('native Megatron commit differs from the pinned baseline')
    require(framework, ['checkout_path', 'source_manifest_sha256', 'bridge_code_sha256', 'native_backend'], 'framework')
    require_true(framework, ['native_source_verified'], 'framework')
    artifact(framework.get('verification_receipt'), 'framework.verification_receipt', prerequisites)
    model = section('model')
    if model.get('variant') != '9G7B_MHA':
        prerequisites.append('selected checkpoint is not the identified 7B MHA variant')
    require(model, ['checkpoint_path', 'checkpoint_manifest_sha256', 'config_sha256', 'tokenizer_manifest_sha256'], 'model')
    if model.get('config_sha256') != CONFIG_SHA:
        prerequisites.append('actual model config does not match the inspected official 7B config')
    count = model.get('parameter_elements_from_actual_tensors')
    if not positive_integer(count) or not 6_000_000_000 <= count <= 9_000_000_000:
        prerequisites.append('actual billion-scale 7B tensor parameter count is unverified')
    require_true(model, ['complete_checkpoint_bytes_verified', 'tensor_structure_and_QKV_bias_verified',
                        'original_checkpoint_loaded_verified', 'megatron_bridge_numerical_equivalence_verified',
                        'seven_b_semantics_without_eight_b_scaling_verified'], 'model')
    artifact(model.get('verification_receipt'), 'model.verification_receipt', prerequisites)
    runtime = section('runtime')
    require(runtime, ['platform', 'container_identifier', 'gpu_model', 'torch_version',
                      'driver_version', 'communication_library_version', 'host_memory_limit_bytes'], 'runtime')
    available, uuids = runtime.get('available_gpus'), runtime.get('device_uuids')
    if not positive_integer(available) or not 2 <= available <= 4096:
        prerequisites.append('at least two allocated GPUs have not been verified')
    if not isinstance(uuids, list) or not all(isinstance(x, str) and x for x in uuids) or len(set(uuids)) < 2:
        prerequisites.append('two distinct allocated GPU UUIDs are required')
    if not positive_integer(runtime.get('host_memory_limit_bytes')):
        prerequisites.append('runtime host memory limit is not a positive byte count')
    require_true(runtime, ['allocation_verified', 'other_active_compute_processes_checked'], 'runtime')
    artifact(runtime.get('verification_receipt'), 'runtime.verification_receipt', prerequisites)
    topology = section('topology')
    world = topology.get('world_size')
    parts = [topology.get(key) for key in ('tp', 'pp', 'cp', 'dp')]
    if not all(positive_integer(value) for value in parts) or not positive_integer(world) or not 2 <= world <= 4096:
        prerequisites.append('a valid at-least-two-rank TP/PP/CP/DP topology is not fixed')
        valid_world = None
    else:
        valid_world = world
        if parts[0] * parts[1] * parts[2] * parts[3] != world:
            prerequisites.append('TP*PP*CP*DP does not match world_size')
        if positive_integer(available) and world > available:
            prerequisites.append('topology exceeds allocated GPU count')
        if 28 % parts[0] or 28 % parts[1] or 3584 % parts[0]:
            prerequisites.append('7B MHA heads/layers do not divide the selected TP/PP topology')
    require(topology, ['interconnect', 'memory_strategy'], 'topology')
    mapping = topology.get('rank_device_host_mapping')
    if not isinstance(mapping, list) or not valid_world or len(mapping) != valid_world:
        prerequisites.append('all-rank physical device/host mapping is missing')
    else:
        ranks, mapped_devices, local_slots = [], [], []
        for item in mapping:
            if not isinstance(item, dict):
                prerequisites.append('invalid rank mapping entry')
                continue
            rank = item.get('rank')
            if not isinstance(rank, int) or isinstance(rank, bool):
                prerequisites.append('rank mapping rank is invalid')
                continue
            ranks.append(rank)
            require(item, ['device_uuid', 'host', 'local_rank'], 'topology.rank_mapping')
            mapped_devices.append(item.get('device_uuid'))
            local_rank, host = item.get('local_rank'), item.get('host')
            if not isinstance(local_rank, int) or isinstance(local_rank, bool) or local_rank < 0 or not isinstance(host, str) or not host:
                prerequisites.append('rank mapping has invalid local_rank/host')
            else:
                local_slots.append((host, local_rank))
        if set(ranks) != set(range(valid_world)) or len(ranks) != valid_world:
            prerequisites.append('rank mapping is incomplete or duplicated')
        if len(set(str(x) for x in mapped_devices)) != valid_world or not isinstance(uuids, list) or any(x not in uuids for x in mapped_devices):
            prerequisites.append('rank mapping does not use distinct verified allocated GPUs')
        if len(set(local_slots)) != valid_world:
            prerequisites.append('rank mapping reuses a host/local_rank slot')
    require_true(topology, ['native_full_step_memory_fit_verified'], 'topology')
    artifact(topology.get('verification_receipt'), 'topology.verification_receipt', prerequisites)
    task = section('task')
    if task.get('objective') != 'continued_pretraining_next_token_on_entire_document' or task.get('answer_only_loss_mask') is not False:
        prerequisites.append('CPT must use full-document next-token loss rather than answer-only SFT loss')
    if task.get('mock_data') is not False or task.get('all_model_parameters_trainable') is not True:
        prerequisites.append('real data and full-parameter training are required')
    if task.get('dataset_original_sha256') != DATA_SHA or task.get('dataset_prepared_sha256') != PREPARED_SHA:
        prerequisites.append('dataset hashes differ from the prepared specified Chinese Alpaca data')
    require(task, ['tokenized_dataset_path', 'tokenized_dataset_manifest_sha256', 'tokenization_code_sha256',
                   'loss_mask_code_sha256', 'precision', 'optimizer', 'optimizer_config_sha256',
                   'activation_recompute', 'offload_strategy', 'EOD_token_ids', 'packing_policy', 'training_argv'], 'task')
    for key in ('sequence_length', 'micro_batch_size', 'global_batch_size', 'gradient_accumulation_steps'):
        if not positive_integer(task.get(key)):
            prerequisites.append('task.' + key + ' is not a positive integer')
    if all(positive_integer(task.get(key)) for key in ('micro_batch_size', 'global_batch_size', 'gradient_accumulation_steps')) and positive_integer(topology.get('dp')):
        if task['global_batch_size'] != task['micro_batch_size'] * topology['dp'] * task['gradient_accumulation_steps']:
            prerequisites.append('global batch does not match micro-batch * DP * gradient accumulation')
    argv = task.get('training_argv')
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv) or any('--mock-data' in x for x in argv if isinstance(x, str)):
        prerequisites.append('actual training argv is missing/invalid or uses mock data')
    require_true(task, ['padding_loss_exclusion_verified',
                       'complete_native_forward_backward_gradient_sync_optimizer_step_verified'], 'task')
    artifact(task.get('verification_receipt'), 'task.verification_receipt', prerequisites)
    capture = section('capture')
    require(capture, ['backend', 'clock_alignment_method', 'collective_identification_rules'], 'capture')
    if capture.get('activities') != ['CPU', 'CUDA']:
        prerequisites.append('both CPU and CUDA device activities are required')
    for key in ('record_shapes', 'with_stack'):
        if not isinstance(capture.get(key), bool):
            prerequisites.append('capture.' + key + ' is not fixed')
    if not positive_integer(capture.get('warmup_steps')):
        prerequisites.append('capture warmup is not fixed')
    require_true(capture, ['window_boundaries_include_full_gpu_work', 'window_complete_training_steps_verified',
                          'no_per_step_synchronization_inserted', 'no_per_operator_synchronization_inserted',
                          'collective_gpu_capture_verified', 'cpu_gpu_clock_correlation_verified',
                          'cross_rank_clock_alignment_verified', 'expected_collective_activity'], 'capture')
    scopes = capture.get('full_scope')
    if not isinstance(scopes, list) or not FULL_SCOPE.issubset(set(x for x in scopes if isinstance(x, str))):
        prerequisites.append('timeline omits data/forward/backward/communication/optimizer/waits')
    artifact(capture.get('clock_alignment_evidence'), 'capture.clock_alignment_evidence', prerequisites)
    formal = section('frozen_formal_benchmark_reference')
    artifact(formal, 'frozen_formal_benchmark_reference', prerequisites)
    if formal.get('profile_timing_is_not_formal_speedup_acceptance') is not True or formal.get('applies_to_later_unprofiled_performance_acceptance') is not True or formal.get('require_120_steps_to_admit_this_tracing_task') is not False:
        prerequisites.append('tracing/performance gate separation has changed')
    windows = capture.get('windows')
    if not isinstance(windows, list) or not 2 <= len(windows) <= 8:
        prerequisites.append('at least two distinct capture windows are required (maximum eight)')
        windows = []
    run_ids, exports, local_dirs, all_paths = [], [], [], []
    for index, window in enumerate(windows):
        prefix = 'capture.windows.' + str(index)
        if not isinstance(window, dict):
            prerequisites.append(prefix + ' is not an object')
            continue
        require(window, ['run_id', 'export_directory', 'local_trace_directory', 'stable_step_ids'], prefix)
        run_ids.append(window.get('run_id'))
        exports.append(posixpath.normpath(str(window.get('export_directory')).replace('\\', '/')))
        local_folder = window.get('local_trace_directory')
        local_dirs.append(str((base_directory / local_folder).resolve()) if isinstance(local_folder, str) else str(local_folder))
        step_ids = window.get('stable_step_ids')
        valid_steps = isinstance(step_ids, list) and 1 <= len(step_ids) <= 256 and all(
            isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in step_ids)
        if not valid_steps or len(set(step_ids)) != len(step_ids):
            prerequisites.append(prefix + '.stable_step_ids is missing/invalid/duplicated')
        selector = window.get('window_selector')
        if not isinstance(selector, dict) or not isinstance(selector.get('name'), str) or not selector['name']:
            prerequisites.append(prefix + ' needs a named complete multi-step window annotation')
        if window.get('stability_verified') is not True:
            delivery.append(prefix + ' is not verified stable after warmup')
        if window.get('dropped_events') != 0 or isinstance(window.get('dropped_events'), bool):
            delivery.append(prefix + ' zero dropped events is not established')
        if window.get('capture_errors') != []:
            delivery.append(prefix + ' has capture errors or missing error observation')
        files = window.get('rank_traces')
        if not isinstance(files, list) or not valid_world or len(files) != valid_world:
            delivery.append(prefix + ' does not contain all-rank traces')
            files = []
        recorded_ranks, actual_json_hashes = [], {}
        for item in files:
            if not isinstance(item, dict):
                delivery.append(prefix + ' trace entry is not an object')
                continue
            rank = item.get('rank')
            if not isinstance(rank, int) or isinstance(rank, bool):
                delivery.append(prefix + ' trace rank is invalid')
                continue
            recorded_ranks.append(rank)
            path = artifact(item, prefix + '.rank-' + str(rank), delivery, read_json=False)
            if path:
                all_paths.append(path)
                folder = window.get('local_trace_directory')
                if not isinstance(folder, str) or not path.is_relative_to((base_directory / folder).resolve()):
                    delivery.append(prefix + ' trace is outside its unique local capture directory')
                if path.name != 'rank-' + str(rank) + '.json.gz':
                    delivery.append(prefix + ' trace filename differs from the pinned native exporter')
                try:
                    trace, source = load_trace(path)
                    actual_json_hashes[rank] = source['decompressed_json_sha256']
                    trace_metadata = {
                        'world_size': valid_world, 'stable_step_ids': step_ids,
                        'gpu_activity_enabled': capture.get('activities') == ['CPU', 'CUDA'],
                        'step_boundaries_include_full_gpu_work': capture.get('step_boundaries_include_full_gpu_work'),
                        'window_boundaries_include_full_gpu_work': capture.get('window_boundaries_include_full_gpu_work'),
                        'window_complete_training_steps_verified': capture.get('window_complete_training_steps_verified'),
                        'collective_gpu_capture_verified': capture.get('collective_gpu_capture_verified'),
                        'expected_collective_activity': capture.get('expected_collective_activity'),
                        'dropped_events': window.get('dropped_events'), 'capture_errors': window.get('capture_errors')}
                    trace_analysis = analyze_rank_trace(trace, rank, step_ids, trace_metadata, window=selector)
                    if trace_analysis['status'] != 'complete_descriptive':
                        delivery.append(prefix + ' actual rank-' + str(rank) + ' trace coverage is incomplete')
                    if trace_analysis['trace_world_size'] is not None and trace_analysis['trace_world_size'] != valid_world:
                        delivery.append(prefix + ' actual trace world_size disagrees with topology')
                except (OSError, ValueError, TypeError) as error:
                    delivery.append(prefix + ' actual trace is invalid: ' + str(error))
        if valid_world and (set(recorded_ranks) != set(range(valid_world)) or len(recorded_ranks) != valid_world):
            delivery.append(prefix + ' rank trace coverage is missing/duplicated')
        analysis = artifact(window.get('analysis_receipt'), prefix + '.analysis_receipt', delivery)
        if analysis:
            if analysis.get('schema') != 'jiuge-kineto-analysis-v1' or analysis.get('status') != 'complete_descriptive':
                delivery.append(prefix + ' offline device/step/communication analysis is incomplete')
            if analysis.get('analysis_scope') != 'complete_capture_window':
                delivery.append(prefix + ' analysis does not cover the complete multi-step activity window')
            if analysis.get('selected_step_ids') != step_ids:
                delivery.append(prefix + ' analysis selected steps disagree with capture window')
            analyzed = analysis.get('ranks')
            valid_analyzed = isinstance(analyzed, list) and all(isinstance(x, dict) and
                isinstance(x.get('rank'), int) and not isinstance(x.get('rank'), bool) for x in analyzed)
            if not valid_analyzed or not valid_world or {x['rank'] for x in analyzed} != set(range(valid_world)) or len(analyzed) != valid_world:
                delivery.append(prefix + ' analysis lacks all ranks')
            elif any(x.get('status') != 'complete_descriptive' for x in analyzed if isinstance(x, dict)):
                delivery.append(prefix + ' analysis contains incomplete ranks')
            elif any(not isinstance(x.get('capture_window'), dict) or
                     x['capture_window'].get('selector') != selector for x in analyzed):
                delivery.append(prefix + ' analysis window selector disagrees with capture contract')
            sources = analysis.get('sources')
            if not isinstance(sources, list) or not all(isinstance(x, dict) and
                isinstance(x.get('rank'), int) and not isinstance(x.get('rank'), bool) for x in sources):
                delivery.append(prefix + ' analysis source byte identity is unavailable')
            else:
                reported_hashes = {x['rank']: x.get('decompressed_json_sha256') for x in sources}
                if reported_hashes != actual_json_hashes or not valid_world or len(sources) != valid_world:
                    delivery.append(prefix + ' analysis was not produced from these exact rank trace bytes')
    if any(not isinstance(x, str) or not x for x in run_ids) or len(set(str(x) for x in run_ids)) != len(run_ids):
        prerequisites.append('capture run_id must be unique and nonempty per window')
    if any(not isinstance(x, str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}', x) for x in run_ids):
        prerequisites.append('capture run_id must be a plain identifier of at most 128 characters')
    if len(set(exports)) != len(exports) or len(set(local_dirs)) != len(local_dirs):
        prerequisites.append('capture export/local directories must be distinct; native rank files would be overwritten')
    if len(set(all_paths)) != len(all_paths):
        delivery.append('trace file path is reused across windows or ranks')
    prerequisites, delivery = sorted(set(prerequisites)), sorted(set(delivery))
    return {'schema': 'jiuge-7b-cpt-tracing-readiness-v1',
            'capture_prerequisites_ready': not prerequisites,
            'tracing_delivery_ready': not prerequisites and not delivery,
            'prerequisite_issues': prerequisites, 'delivery_issues': delivery,
            'artifact_files_checked': len(checked), 'artifact_bytes_hashed': consumed,
            'checks_gpu_math_or_weight_tensors': False, 'starts_training': False,
            'formal_performance_passed': False, 'multi_node_validated': False,
            'note': 'CPU integrity/configuration review of producer receipts; no GPU numerical or performance proof.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=Path(__file__).with_name('tracing.contract.template.json'))
    parser.add_argument('--phase', choices=['preparation', 'delivery'], default='delivery')
    parser.add_argument('--max-artifact-mib', type=int, default=1024)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.contract.stat().st_size > 4 * 1024 * 1024 or not 1 <= args.max_artifact_mib <= 8192:
        parser.error('contract or artifact byte limit exceeds supported bounds')
    result = check_contract(json.loads(args.contract.read_text(encoding='utf-8')),
                            args.contract.parent, args.max_artifact_mib * 1024 * 1024)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ready = result['capture_prerequisites_ready'] if args.phase == 'preparation' else result['tracing_delivery_ready']
    return 0 if ready else 2


if __name__ == '__main__':
    raise SystemExit(main())
