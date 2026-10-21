import copy
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from preflight import check_contract, resident_adamw_estimate
from resource_probe import resource_gate
from stage_checkpoint import member_path, stage, OFFICIAL_URL


def workspace_temporary_directory():
    workspace = Path(__file__).resolve().parents[2]
    test_root = Path(__file__).resolve().parent / 'results/test-temporary'
    if not test_root.resolve().is_relative_to(workspace):
        raise ValueError('Temporary test root escaped workspace')
    test_root.mkdir(parents=True, exist_ok=True)
    folder = tempfile.TemporaryDirectory(dir=test_root)
    if not Path(folder.name).resolve().is_relative_to(test_root.resolve()):
        raise ValueError('Temporary cleanup target escaped test directory')
    return folder


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = json.loads(Path(__file__).with_name('contract.template.json').read_text(encoding='utf-8'))

    def test_incomplete_real_checkpoint_blocks(self):
        result = check_contract(self.template)
        self.assertFalse(result['configuration_ready'])
        self.assertIn('checkpoint conversion/loading equivalence is unverified', result['issues'])

    def test_relaxed_tolerance_rejected(self):
        data = copy.deepcopy(self.template)
        data['benchmark']['atol'] = .01
        self.assertIn('benchmark differs from the previous frozen standard', check_contract(data)['issues'])

    def test_changed_timer_rejected(self):
        data = copy.deepcopy(self.template)
        data['benchmark']['formal_timer'] = 'triton.testing.do_bench'
        self.assertIn('benchmark differs from the previous frozen standard', check_contract(data)['issues'])

    def test_mock_data_cannot_be_real_model_acceptance(self):
        data = copy.deepcopy(self.template)
        data['tasks']['sft']['training_argv'] = ['--mock-data']
        self.assertIn('sft uses mock data', check_contract(data)['issues'])

    def test_oversized_pilot_topology_blocks(self):
        data = copy.deepcopy(self.template)
        data['topology'].update(pilot_tp=8, pilot_pp=4, pilot_dp=2, available_gpus=2)
        self.assertIn('pilot topology exceeds verified available GPUs', check_contract(data)['issues'])

    def test_adamw_estimate_not_inference_weight_memory(self):
        result = resident_adamw_estimate(8_000_000_000)
        self.assertEqual(result['total_bytes'], 144_000_000_000)
        self.assertFalse(result['measured_peak_memory'])

    def test_pinned_native_kv_replication_allows_tp8_groups2(self):
        data = copy.deepcopy(self.template)
        data['model'].update(hidden_size=4096, num_attention_heads=32, num_query_groups=2)
        data['topology'].update(pilot_tp=8, pilot_pp=4, pilot_dp=2, available_gpus=64)
        self.assertNotIn('head/GQA partitioning needs explicit Megatron compatibility proof', check_contract(data)['issues'])

    def test_live_foreign_compute_blocks_even_with_idle_cards(self):
        gpu = {'returncode': 0, 'stdout': '0MiB / 32768MiB | 0%\nNo running processes found'}
        processes = {'returncode': 0, 'stdout': 'PID PPID USER ETIMES PCPU PMEM STAT COMM\n42 1 root 1 0.0 0.0 S python'}
        self.assertFalse(resource_gate(gpu, processes, own_pid=99)['passed'])

    def test_nonexecuting_zombie_does_not_block(self):
        gpu = {'returncode': 0, 'stdout': '0MiB / 32768MiB | 0%\nNo running processes found'}
        processes = {'returncode': 0, 'stdout': 'PID PPID USER ETIMES PCPU PMEM STAT COMM\n42 1 root 1 0.0 0.0 Z python'}
        self.assertTrue(resource_gate(gpu, processes, own_pid=99)['passed'])

    def test_failed_gpu_observation_cannot_admit(self):
        self.assertFalse(resource_gate({'returncode': 1}, {'returncode': 0}, 99)['passed'])

    def test_checkpoint_path_escape_rejected(self):
        with workspace_temporary_directory() as folder:
            with self.assertRaises(ValueError):
                member_path(Path(folder), {'name': '9g_8b_thinking/../../outside'})

    def test_timeout_resume_preserves_exact_checkpoint_bytes(self):
        payload = b'0123456789abcdef'
        manifest = {'url': OFFICIAL_URL, 'status': 'metadata_read', 'archive_bytes': 528,
                    'members': [{'name': '9g_8b_thinking/config.json', 'bytes': 16, 'data_offset': 512}],
                    'metadata_files': {'config.json': {'sha256': hashlib.sha256(payload).hexdigest()}}}
        requests = []

        class Reply:
            def __init__(self, status, headers, data=b'', fail_after_first=False):
                self.status, self.headers, self.data = status, headers, data
                self.fail_after_first = fail_after_first
                self.calls = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, count):
                self.calls += 1
                if self.fail_after_first and self.calls > 1:
                    raise TimeoutError('interrupted payload')
                result, self.data = self.data[:count], self.data[count:]
                return result

        def urlopen(request, timeout):
            requests.append(request)
            if request.get_method() == 'HEAD':
                return Reply(200, {'Content-Length': '528', 'ETag': 'fixed'})
            if len(requests) == 2:
                return Reply(206, {'Content-Range': 'bytes 512-527/528', 'ETag': 'fixed'},
                             payload[:8], fail_after_first=True)
            self.assertEqual(request.get_header('Range'), 'bytes=520-527')
            return Reply(206, {'Content-Range': 'bytes 520-527/528', 'ETag': 'fixed'}, payload[8:])

        with workspace_temporary_directory() as folder:
            root = Path(folder)
            with patch('stage_checkpoint.urllib.request.urlopen', side_effect=urlopen), patch('stage_checkpoint.time.sleep'):
                stage(manifest, root / 'checkpoint', root / 'receipt.json')
            self.assertEqual((root / 'checkpoint/9g_8b_thinking/config.json').read_bytes(), payload)
            receipt = json.loads((root / 'receipt.json').read_text())
            self.assertEqual(receipt['status'], 'complete')
            self.assertEqual(len(receipt['network_retries']), 1)


if __name__ == '__main__':
    unittest.main()
