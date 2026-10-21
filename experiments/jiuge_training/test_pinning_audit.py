"""Checks that full-state audit raw chunks survive BF16 and reject corruption."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from pinning_full_audit import FullAudit


class PinningAuditTest(unittest.TestCase):
    def test_shared_bytes_keep_each_dtype_element_count(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            native = FullAudit(root, 'native', 0)
            first = native.tensor('gradient', torch.zeros(4, dtype=torch.float32))
            second = native.tensor('parameter', torch.zeros(8, dtype=torch.bfloat16))
            native.close()
            self.assertEqual(first['chunks'][0]['sha256'], second['chunks'][0]['sha256'])
            self.assertEqual(first['chunks'][0]['numel'], 4)
            self.assertEqual(second['chunks'][0]['numel'], 8)

    def test_strided_bf16_and_tolerant_comparison(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            value = torch.arange(30).bfloat16().reshape(5, 6)[:, ::2]
            native = FullAudit(root, 'native', 0)
            native.tensor('parameter', value)
            native.flush()
            native.close()
            candidate = FullAudit(root, 'optimized', 0)
            row = candidate.tensor('parameter', value.clone())
            candidate.close()
            self.assertTrue(row['passed'])
            self.assertEqual(row['max_abs'], 0.0)

    def test_numerical_error_and_nan_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            native = FullAudit(root, 'native', 0)
            native.tensor('gradient', torch.tensor([1., 2.]))
            native.flush()
            native.close()
            control = FullAudit(root, 'control', 0)
            self.assertFalse(control.tensor('gradient', torch.tensor([1., 3.]))['passed'])
            control.close()
            candidate = FullAudit(root, 'optimized', 0)
            self.assertFalse(candidate.tensor('gradient', torch.tensor([1., float('nan')]))['passed'])
            candidate.close()

    def test_corrupted_reference_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            native = FullAudit(root, 'native', 0)
            row = native.tensor('master', torch.tensor([1., 2.]))
            native.flush()
            native.close()
            chunk = row['chunks'][0]
            with (root / chunk['file']).open('r+b') as stream:
                stream.seek(chunk['offset'])
                import zlib
                stream.write(zlib.compress(b'wrong'))
            candidate = FullAudit(root, 'optimized', 0)
            with self.assertRaisesRegex(ValueError, 'corrupted'):
                candidate.tensor('master', torch.tensor([1., 2.]))
            candidate.close()


if __name__ == '__main__':
    unittest.main()
