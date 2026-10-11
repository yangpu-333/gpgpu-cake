"""Resource guards must block live computation and never stop other jobs."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch, MagicMock

spec = importlib.util.spec_from_file_location('resource_guard_v2', Path(__file__).with_name('browser_formal_retry01_v2.py'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
GPU_IDLE = '116MiB / 32768MiB | 0%\n116MiB / 32768MiB | 0%\nNo running processes found'

class ResourceGuards(unittest.TestCase):
    def check(self, rows, gpu=GPU_IDLE):
        port = MagicMock()
        port.__enter__.return_value.connect_ex.return_value = 111
        responses = [subprocess.CompletedProcess([], 0, gpu, ''),
                     subprocess.CompletedProcess([], 0, 'PID PPID USER ELAPSED CPU MEM STAT COMMAND\n' + rows, '')]
        with patch.object(guard.subprocess, 'run', side_effect=responses), patch.object(guard.socket, 'socket', return_value=port):
            return guard.occupancy()

    def test_live_python_blocks_even_with_low_cpu(self):
        self.assertFalse(self.check('999999 1 root 20 0.0 0.0 S python\n')['passed'])

    def test_live_unrecognized_compute_blocks(self):
        self.assertFalse(self.check('999999 1 root 20 45.6 0.0 R pt_main_thread\n')['passed'])

    def test_zombie_cannot_compute(self):
        self.assertTrue(self.check('999999 1 root 20 45.6 0.0 Z pt_main_thread\n')['passed'])

    def test_busy_gpu_blocks_even_without_cpu_process(self):
        gpu = '2048MiB / 32768MiB | 20%\n116MiB / 32768MiB | 0%'
        self.assertFalse(self.check('', gpu)['passed'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
