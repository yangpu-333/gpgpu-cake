"""Synthetic evidence fixtures exercise gates, never prove real Jiuge training."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_trace import analyze_traces, load_trace
from check_tracing_readiness import check_contract


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def synthetic_trace(rank, window_name='jiuge/capture/synthetic-window'):
    def event(name, start, duration, category):
        return {'ph': 'X', 'name': name, 'ts': start, 'dur': duration, 'cat': category,
                'pid': 0 if category == 'kernel' else 900, 'tid': 1,
                'args': {'device': 0, 'stream': 1}}
    return {'distributedInfo': {'rank': rank, 'world_size': 2}, 'traceEvents': [
        event(window_name, 95, 110, 'user_annotation'),
        event('ProfilerStep#20', 100, 100, 'user_annotation'),
        event('synthetic_gemm', 110, 40, 'kernel'),
        event('ncclDevKernel_AllReduce', 140, 40, 'kernel')]}


class TracingReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent
        cls.template = json.loads((cls.root / 'tracing.contract.template.json').read_text(encoding='utf-8'))

    def temporary_folder(self):
        root = self.root / 'results/test-tracing-readiness-temporary'
        root.mkdir(parents=True, exist_ok=True)
        result = tempfile.TemporaryDirectory(dir=root)
        self.assertTrue(Path(result.name).resolve().is_relative_to(root.resolve()))
        self.addCleanup(result.cleanup)
        return Path(result.name)

    def prepared_fixture(self, folder, captures=True):
        data = copy.deepcopy(self.template)
        proof = write_json(folder / 'synthetic-proof.json', {'synthetic_unit_test_fixture': True})
        for section in ('framework', 'model', 'runtime', 'topology', 'task'):
            data[section]['verification_receipt'] = proof.copy()
        data['framework'].update(checkout_path='/synthetic/source', source_manifest_sha256='a' * 64,
                                  bridge_code_sha256='b' * 64, native_backend='synthetic-test', native_source_verified=True)
        data['model'].update(checkpoint_path='/synthetic/model', checkpoint_manifest_sha256='c' * 64,
                            config_sha256=data['reference']['config_sha256'], tokenizer_manifest_sha256='d' * 64,
                            parameter_elements_from_actual_tensors=7668831744)
        for key in list(data['model']):
            if key.endswith('_verified'):
                data['model'][key] = True
        data['runtime'].update(platform='synthetic-test', container_identifier='synthetic-test',
                              gpu_model='synthetic-test', device_uuids=['synthetic-gpu-0', 'synthetic-gpu-1'],
                              available_gpus=2, torch_version='synthetic-test', driver_version='synthetic-test',
                              communication_library_version='synthetic-test', host_memory_limit_bytes=16 * 2 ** 30,
                              allocation_verified=True, other_active_compute_processes_checked=True)
        data['topology'].update(tp=2, pp=1, cp=1, dp=1, world_size=2, interconnect='synthetic-test',
                               memory_strategy='synthetic-test', native_full_step_memory_fit_verified=True,
                               rank_device_host_mapping=[
                                   {'rank': 0, 'host': 'synthetic-host', 'device_uuid': 'synthetic-gpu-0', 'local_rank': 0},
                                   {'rank': 1, 'host': 'synthetic-host', 'device_uuid': 'synthetic-gpu-1', 'local_rank': 1}])
        data['task'].update(tokenized_dataset_path='/synthetic/data', tokenized_dataset_manifest_sha256='e' * 64,
                            tokenization_code_sha256='f' * 64, loss_mask_code_sha256='1' * 64,
                            sequence_length=128, micro_batch_size=1, global_batch_size=2,
                            gradient_accumulation_steps=2, precision='fp16', optimizer='AdamW',
                            optimizer_config_sha256='2' * 64, activation_recompute='full', offload_strategy='synthetic-test',
                            EOD_token_ids=[2], packing_policy='synthetic-test', padding_loss_exclusion_verified=True,
                            training_argv=['python', 'pretrain_gpt.py', '--train-iters', '24'],
                            complete_native_forward_backward_gradient_sync_optimizer_step_verified=True)
        data['capture'].update(backend='torch.profiler', record_shapes=False, with_stack=False, warmup_steps=10,
                              step_boundaries_include_full_gpu_work=False,
                              window_boundaries_include_full_gpu_work=True,
                              window_complete_training_steps_verified=True, collective_gpu_capture_verified=True,
                              collective_identification_rules=['ncclDevKernel'], cpu_gpu_clock_correlation_verified=True,
                              cross_rank_clock_alignment_verified=True, clock_alignment_method='synthetic-test',
                              clock_alignment_evidence=proof.copy())
        formal = self.root / data['frozen_formal_benchmark_reference']['path']
        data['frozen_formal_benchmark_reference']['path'] = str(formal.resolve())
        for index, window in enumerate(data['capture']['windows']):
            target = folder / ('window-' + str(index))
            window.update(run_id='synthetic-run-' + str(index), export_directory='/synthetic/window-' + str(index),
                          local_trace_directory=str(target), stable_step_ids=[20],
                          window_selector={'name': 'jiuge/capture/synthetic-run-' + str(index)})
            if not captures:
                continue
            window.update(stability_verified=True, dropped_events=0, capture_errors=[])
            traces, sources = {}, []
            for rank in range(2):
                path = target / ('rank-' + str(rank) + '.json.gz')
                path.parent.mkdir(parents=True, exist_ok=True)
                traces[rank] = synthetic_trace(rank, window['window_selector']['name'])
                with gzip.open(path, 'wt', encoding='utf-8') as stream:
                    json.dump(traces[rank], stream)
                window['rank_traces'].append({'rank': rank, 'path': str(path),
                                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
                _, source = load_trace(path)
                source['rank'] = rank
                sources.append(source)
            metadata = {'world_size': 2, 'stable_step_ids': [20], 'gpu_activity_enabled': True,
                        'step_boundaries_include_full_gpu_work': False,
                        'window_boundaries_include_full_gpu_work': True,
                        'window_complete_training_steps_verified': True, 'collective_gpu_capture_verified': True,
                        'expected_collective_activity': True, 'dropped_events': 0, 'capture_errors': []}
            analysis = analyze_traces(traces, [20], metadata, window=window['window_selector'])
            analysis['sources'] = sources
            window['analysis_receipt'] = write_json(target / 'analysis.json', analysis)
        return data

    def test_actual_template_blocks_without_claiming_gpu_work(self):
        result = check_contract(self.template, self.root)
        self.assertFalse(result['capture_prerequisites_ready'])
        self.assertFalse(result['tracing_delivery_ready'])
        self.assertFalse(result['checks_gpu_math_or_weight_tensors'])
        self.assertFalse(result['formal_performance_passed'])

    def test_two_windows_all_ranks_integrity_complete_fixture(self):
        folder = self.temporary_folder()
        result = check_contract(self.prepared_fixture(folder), folder)
        self.assertTrue(result['capture_prerequisites_ready'], result['prerequisite_issues'])
        self.assertTrue(result['tracing_delivery_ready'], result['delivery_issues'])

    def test_preparation_gate_does_not_require_120_steps_or_team_tasks(self):
        folder = self.temporary_folder()
        data = self.prepared_fixture(folder, captures=False)
        result = check_contract(data, folder)
        self.assertTrue(result['capture_prerequisites_ready'], result['prerequisite_issues'])
        self.assertFalse(result['tracing_delivery_ready'])
        self.assertEqual(data['team_coverage'][0]['continued_pretraining'], 'not_run')

    def test_single_gpu_or_unverified_full_step_cannot_admit(self):
        folder = self.temporary_folder()
        data = self.prepared_fixture(folder, captures=False)
        data['runtime']['available_gpus'] = 1
        data['runtime']['device_uuids'] = ['synthetic-gpu-0']
        data['task']['complete_native_forward_backward_gradient_sync_optimizer_step_verified'] = False
        result = check_contract(data, folder)
        self.assertFalse(result['capture_prerequisites_ready'])
        self.assertTrue(any('at least two' in issue for issue in result['prerequisite_issues']))

    def test_sft_mask_wrong_model_and_unverified_clock_cannot_admit(self):
        folder = self.temporary_folder()
        data = self.prepared_fixture(folder, captures=False)
        data['task']['answer_only_loss_mask'] = True
        data['model']['variant'] = '9g_8b_thinking'
        data['capture']['cross_rank_clock_alignment_verified'] = False
        self.assertFalse(check_contract(data, folder)['capture_prerequisites_ready'])

    def test_missing_rank_and_reused_export_directory_block_delivery(self):
        folder = self.temporary_folder()
        data = self.prepared_fixture(folder)
        data['capture']['windows'][1]['export_directory'] = data['capture']['windows'][0]['export_directory']
        data['capture']['windows'][0]['rank_traces'].pop()
        result = check_contract(data, folder)
        self.assertFalse(result['capture_prerequisites_ready'])
        self.assertFalse(result['tracing_delivery_ready'])

    def test_changed_trace_bytes_and_wrong_analysis_provenance_block(self):
        folder = self.temporary_folder()
        data = self.prepared_fixture(folder)
        window = data['capture']['windows'][0]
        path = Path(window['rank_traces'][0]['path'])
        path.write_bytes(path.read_bytes() + b'changed')
        result = check_contract(data, folder)
        self.assertFalse(result['tracing_delivery_ready'])
        self.assertTrue(any('SHA256 mismatch' in issue for issue in result['delivery_issues']))

    def test_claimed_complete_analysis_cannot_hide_missing_device_communication(self):
        folder = self.temporary_folder()
        data = self.prepared_fixture(folder)
        item = data['capture']['windows'][0]['rank_traces'][0]
        path = Path(item['path'])
        trace = synthetic_trace(0, data['capture']['windows'][0]['window_selector']['name'])
        trace['traceEvents'] = [event for event in trace['traceEvents'] if 'nccl' not in event['name']]
        with gzip.open(path, 'wt', encoding='utf-8') as stream:
            json.dump(trace, stream)
        item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = check_contract(data, folder)
        self.assertFalse(result['tracing_delivery_ready'])
        self.assertTrue(any('trace coverage is incomplete' in issue for issue in result['delivery_issues']))


if __name__ == '__main__':
    unittest.main()
