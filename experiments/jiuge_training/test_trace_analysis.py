"""Synthetic fixtures test accounting/coverage only; never real-model evidence."""
import gzip
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_trace import (analyze_rank_trace, analyze_traces, intersection_length,
                           interval_length, load_trace, main, union_intervals)


def event(name, start, duration, category='kernel', device=0, stream=1):
    return {'ph': 'X', 'cat': category, 'name': name, 'ts': start, 'dur': duration,
            'pid': 900 if category == 'user_annotation' else device,
            'tid': stream, 'args': {'device': device, 'stream': stream}}


def capture_metadata():
    return {'world_size': 2, 'stable_step_ids': [20], 'gpu_activity_enabled': True,
            'step_boundaries_include_full_gpu_work': True,
            'collective_gpu_capture_verified': True, 'expected_collective_activity': True,
            'dropped_events': 0, 'capture_errors': []}


def fixture(rank=0):
    return {'distributedInfo': {'rank': rank, 'world_size': 2}, 'displayTimeUnit': 'ms',
            'traceEvents': [event('ProfilerStep#20', 100, 100, 'user_annotation'),
                            event('gemm', 110, 40), event('gemm2', 120, 40, stream=2),
                            event('ncclDevKernel_AllReduce', 140, 40, stream=3),
                            event('Memcpy HtoD', 185, 5, 'gpu_memcpy'),
                            event('aten::linear', 100, 95, 'cpu_op')]}


class TraceAnalysisTests(unittest.TestCase):
    def test_gpu_annotation_does_not_fill_idle_gaps_or_duplicate_device_work(self):
        trace = fixture()
        trace['traceEvents'].append(event('ProfilerStep#20', 100, 100, 'gpu_user_annotation'))
        result = analyze_rank_trace(trace, 0, [20], capture_metadata())
        self.assertEqual(result['status'], 'complete_descriptive')
        data = result['steps'][0]['devices'][0]
        self.assertEqual(data['gpu_idle_proxy_us'], 25)
        self.assertEqual(data['interval_union_us']['unknown_gpu'], 0)
        self.assertEqual(data['kernel_work_sum_us']['compute'], 80)

    def test_async_gpu_crosses_steps_but_complete_enclosing_window_is_valid(self):
        trace = {'distributedInfo': {'rank': 0, 'world_size': 2}, 'traceEvents': [
            event('jiuge/capture/synthetic-window', 95, 125, 'user_annotation'),
            event('ProfilerStep#20', 100, 50, 'user_annotation'),
            event('ProfilerStep#21', 150, 50, 'user_annotation'),
            event('gemm', 110, 50), event('gemm-next', 160, 50),
            event('ncclDevKernel_AllReduce', 140, 50, stream=2)]}
        metadata = capture_metadata()
        metadata.update(stable_step_ids=[20, 21], step_boundaries_include_full_gpu_work=False,
                        window_boundaries_include_full_gpu_work=True,
                        window_complete_training_steps_verified=True)
        result = analyze_rank_trace(trace, 0, [20, 21], metadata,
                                    window={'name': 'jiuge/capture/synthetic-window'})
        self.assertEqual(result['status'], 'complete_descriptive')
        self.assertFalse(result['per_step_attribution_complete'])
        self.assertTrue(all(x['status'] == 'incomplete' for x in result['steps']))
        whole = result['capture_window']['devices'][0]
        self.assertEqual(whole['kernel_work_sum_us']['compute'], 100)
        self.assertEqual(whole['compute_collective_overlap_us'], 50)
        self.assertEqual(whole['gpu_idle_proxy_us'], 25)
        # Equivalent explicit ts range is supported without altering the workload.
        ranged = analyze_rank_trace(trace, 0, [20, 21], metadata,
                                     window={'start_ts_us': 95, 'end_ts_us': 220})
        self.assertEqual(ranged['status'], 'complete_descriptive')
        trace['traceEvents'].append(event('out-of-window-tail', 215, 10))
        self.assertEqual(analyze_rank_trace(trace, 0, [20, 21], metadata,
            window={'name': 'jiuge/capture/synthetic-window'})['status'], 'incomplete')

    def test_interval_union_and_intersection(self):
        self.assertEqual(union_intervals([(1, 5), (3, 8), (8, 10), (20, 20)]), [(1, 10)])
        self.assertEqual(interval_length([(1, 5), (3, 8)]), 7)
        self.assertEqual(intersection_length([(0, 5), (3, 9)], [(4, 6), (8, 10)]), 3)

    def test_work_sum_differs_from_union_cpu_nested_time_excluded(self):
        result = analyze_rank_trace(fixture(), 0, [20], capture_metadata())
        self.assertEqual(result['status'], 'complete_descriptive')
        data = result['steps'][0]['devices'][0]
        self.assertEqual(data['interval_union_us']['compute'], 50)
        self.assertEqual(data['kernel_work_sum_us']['compute'], 80)
        self.assertEqual(data['compute_collective_overlap_us'], 20)
        self.assertEqual(data['collective_nonoverlap_with_compute_proxy_us'], 20)
        self.assertEqual(data['gpu_idle_proxy_us'], 25)
        self.assertEqual(data['window_us'], 100)  # displayTimeUnit must not rescale ts/dur.
        self.assertFalse(result['critical_path_proven'])
        self.assertFalse(result['formal_performance_passed'])

    def test_missing_metadata_cannot_be_complete(self):
        self.assertEqual(analyze_rank_trace(fixture(), 0, [20])['status'], 'incomplete')

    def test_cpu_collective_is_not_device_communication(self):
        trace = fixture()
        trace['traceEvents'] = [item for item in trace['traceEvents'] if 'nccl' not in item['name']]
        trace['traceEvents'].append(event('nccl:all_reduce', 140, 40, 'user_annotation'))
        result = analyze_rank_trace(trace, 0, [20], capture_metadata())
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['steps'][0]['devices'][0]['interval_union_us']['collective'], 0)

    def test_missing_boundary_and_gpu_events(self):
        trace = {'traceEvents': [event('ProfilerStep#21', 100, 100, 'user_annotation')]}
        self.assertEqual(analyze_rank_trace(trace, 0, [20], capture_metadata())['steps'], [])
        trace['traceEvents'][0]['name'] = 'ProfilerStep#20'
        result = analyze_rank_trace(trace, 0, [20], capture_metadata())
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['steps'][0]['devices'], [])

    def test_spilling_gpu_work_flags_boundary_coverage(self):
        trace = fixture()
        trace['traceEvents'].append(event('tail', 195, 20))
        result = analyze_rank_trace(trace, 0, [20], capture_metadata())
        self.assertEqual(result['status'], 'incomplete')
        self.assertIn('GPU activity crosses a selected step boundary', result['steps'][0]['issues'])

    def test_devices_and_ranks_are_never_combined(self):
        trace = fixture()
        trace['traceEvents'].append(event('other_device', 110, 40, device=1))
        result = analyze_rank_trace(trace, 0, [20], capture_metadata())
        self.assertEqual(len(result['steps'][0]['devices']), 2)
        combined = analyze_traces({0: fixture(0), 1: fixture(1)}, [20], capture_metadata())
        self.assertEqual(combined['status'], 'complete_descriptive')
        self.assertEqual(len(combined['ranks']), 2)
        self.assertFalse(combined['cross_rank_or_cross_device_time_sum_reported'])
        self.assertEqual(analyze_traces({0: fixture(0)}, [20], capture_metadata())['status'], 'incomplete')

    def test_missing_device_unknown_gpu_and_dropped_events_block_coverage(self):
        trace = fixture()
        trace['traceEvents'][1].pop('args')
        trace['traceEvents'].append(event('unrecognized', 150, 10, 'gpu_unrecognized'))
        metadata = capture_metadata()
        metadata['dropped_events'] = 1
        result = analyze_rank_trace(trace, 0, [20], metadata)
        self.assertEqual(result['status'], 'incomplete')
        self.assertTrue(any('physical-device' in issue for issue in result['issues']))
        self.assertTrue(any('unknown categories' in issue for issue in result['issues']))

    def test_duplicate_boundaries_and_invalid_timestamp_block_coverage(self):
        trace = fixture()
        trace['traceEvents'].append(event('ProfilerStep#20', 100, 100, 'user_annotation'))
        trace['traceEvents'].append(event('broken', float('nan'), 10))
        result = analyze_rank_trace(trace, 0, [20], capture_metadata())
        self.assertEqual(result['status'], 'incomplete')

    def test_bounded_json_gzip_loading(self):
        # Resolve and constrain the only recursive cleanup target to this workspace.
        root = Path(__file__).resolve().parent / 'results/test-trace-temporary'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root.resolve()))
            normal = Path(folder) / 'trace.json'
            compressed = Path(folder) / 'trace.json.gz'
            payload = json.dumps(fixture()).encode('utf-8')
            normal.write_bytes(payload)
            with gzip.open(compressed, 'wb') as stream:
                stream.write(payload)
            self.assertEqual(load_trace(normal)[0], load_trace(compressed)[0])
            self.assertEqual(load_trace(normal)[1]['decompressed_json_sha256'],
                             load_trace(compressed)[1]['decompressed_json_sha256'])
            with self.assertRaises(ValueError):
                load_trace(compressed, len(payload) - 1)
            with self.assertRaises(ValueError):
                analyze_rank_trace(fixture(), 0, [20], capture_metadata(), max_events=1)

    def test_cli_two_rank_machine_readable_output_and_missing_rank(self):
        root = Path(__file__).resolve().parent / 'results/test-trace-temporary'
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(root.resolve()))
            paths = [Path(folder) / 'rank0.json', Path(folder) / 'rank1.json.gz']
            paths[0].write_text(json.dumps(fixture(0)), encoding='utf-8')
            with gzip.open(paths[1], 'wt', encoding='utf-8') as stream:
                json.dump(fixture(1), stream)
            metadata = Path(folder) / 'capture.json'
            metadata.write_text(json.dumps(capture_metadata()), encoding='utf-8')
            output = Path(folder) / 'analysis.json'
            command = ['analyze_trace', '--trace', '0=' + str(paths[0]),
                       '--trace', '1=' + str(paths[1]), '--steps', '20',
                       '--capture-metadata', str(metadata), '--output', str(output)]
            with patch('sys.argv', command), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
            result = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(result['status'], 'complete_descriptive')
            self.assertFalse(result['real_jiuge_training_verified'])
            self.assertFalse(result['formal_performance_passed'])
            self.assertEqual(len(result['sources']), 2)
            incomplete = Path(folder) / 'missing-rank.json'
            command = ['analyze_trace', '--trace', '0=' + str(paths[0]), '--steps', '20',
                       '--capture-metadata', str(metadata), '--output', str(incomplete)]
            with patch('sys.argv', command), redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 2)
            self.assertEqual(json.loads(incomplete.read_text(encoding='utf-8'))['status'], 'incomplete')


if __name__ == '__main__':
    unittest.main()
