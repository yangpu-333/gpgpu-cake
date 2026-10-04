"""Protect archived acceptance gates using real small stage29/001 evidence.

Each test edits only its temporary copy. This suite uses the CPU receipt reader,
never imports Torch/candidates, and does not repeat GPU or model API work.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
RAW = HERE.parent / 'evidence/swiglu-stage29'
READER = HERE / 'verify_receipts.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ignore_payloads(directory, names):
    """Never copy tensor/binary backups or diagnostic payload directories."""
    ignored = {'raw-backup', 'snapshots', 'activation', 'profile', '__pycache__'}
    return [name for name in names if name in ignored or
            Path(name).suffix.lower() in ('.pt', '.pth', '.so', '.tar', '.gz')]


class ArchivedReceiptAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='swiglu-receipt-tests-')
        cls.root = Path(cls.temporary.name).resolve()
        if not cls.root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
            raise RuntimeError('test workspace must stay inside the system temporary directory')
        cls.template = cls.root / 'template'
        cls.template.mkdir()
        required = ['candidate-001.py', 'probe-001.json', 'probe_candidate.py',
                    'launch_swiglu.py', 'launch_swiglu-02.py', 'run_candidate_audit.py',
                    'run_performance.py', 'audit-summary-main-001-complete.json',
                    'performance-main-001-complete.json']
        for name in required:
            shutil.copy2(RAW / name, cls.template / name)
        (cls.template / 'contracts').mkdir()
        shutil.copy2(RAW / 'contracts/main.json', cls.template / 'contracts/main.json')
        summary = read(RAW / 'audit-summary-main-001-complete.json')
        for comparison in summary['comparisons'].values():
            name = PurePosixPath(comparison['path']).name
            shutil.copy2(RAW / name, cls.template / name)
        folders = {'main-native-audit', 'main-control-audit-001-complete',
                   'main-optimized-audit-001-complete'}
        performance = read(RAW / 'performance-main-001-complete.json')
        for pair in performance['pairs']:
            folders.update(value['case'] for value in pair['arms'].values())
        for name in sorted(folders):
            shutil.copytree(RAW / name, cls.template / name, ignore=ignore_payloads)
        # The declared backup inventory is deliberately omitted: this fixture
        # tests portable small receipts and never scans or copies the raw GBs.
        if any(path.suffix.lower() in ('.pt', '.pth', '.so')
               for path in cls.template.rglob('*') if path.is_file()):
            raise RuntimeError('test fixture unexpectedly copied a raw tensor or binary')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.case = self.root / self._testMethodName
        shutil.copytree(self.template, self.case)

    def verify(self):
        environment = os.environ.copy()
        environment['PYTHONUTF8'] = '1'
        result = subprocess.run([sys.executable, str(READER), '--version', '001',
                                 '--base', str(self.case)], env=environment,
                                capture_output=True, text=True, encoding='utf-8', timeout=60)
        try:
            report = json.loads(result.stdout)
        except ValueError as error:
            self.fail('receipt reader did not emit JSON: ' + result.stderr + '\n' + str(error))
        return result.returncode, report

    def test_original_small_archive_passes(self):
        code, report = self.verify()
        self.assertEqual(code, 0, report)
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['performance']['timed_processes'], 9)
        self.assertFalse(report['performance']['promoted'])
        self.assertFalse(report['gpu_rerun'])

    def test_false_promotion_is_rejected(self):
        path = self.case / 'performance-main-001-complete.json'
        performance = read(path)
        performance.update(promoted=True, decision='promoted')
        write(path, performance)
        code, report = self.verify()
        self.assertNotEqual(code, 0)
        self.assertEqual(report['status'], 'rejected')
        self.assertIn('promotion contradicts', report['error'])

    def test_faster_summary_with_unchanged_raw_log_is_rejected(self):
        path = self.case / 'performance-main-001-complete.json'
        performance = read(path)
        performance['pairs'][0]['arms']['optimized']['mean_iteration_ms'] /= 2
        write(path, performance)
        code, report = self.verify()
        self.assertNotEqual(code, 0)
        self.assertEqual(report['status'], 'rejected')
        self.assertIn('raw-log mean differs', report['error'])

    def test_failed_numerical_gate_with_rehashed_summary_is_rejected(self):
        summary_path = self.case / 'audit-summary-main-001-complete.json'
        summary = read(summary_path)
        item = summary['comparisons']['controlled']
        comparison_path = self.case / PurePosixPath(item['path']).name
        comparison = read(comparison_path)
        passed = next(check for check in comparison['checks'] if check['passed'] is True)
        passed['passed'] = False
        write(comparison_path, comparison)
        # Recomputing the public outer SHA cannot repair the actual failed gate.
        item['sha256'] = digest(comparison_path)
        write(summary_path, summary)
        code, report = self.verify()
        self.assertNotEqual(code, 0)
        self.assertEqual(report['status'], 'rejected')
        self.assertIn('actual complete gate list', report['error'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
