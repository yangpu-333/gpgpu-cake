"""Meaningful negative tests for master coverage and the retained timing metric."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import verify_receipts as v


class ArchiveGates(unittest.TestCase):
    def test_missing_master_tensor_is_rejected(self):
        original = v.read
        def modified(path):
            value = original(path)
            if Path(path).name == 'comparison-control-004.json':
                value = copy.deepcopy(value)
                rows = value['tensor_comparisons']
                rows.pop(next(i for i, row in enumerate(rows) if row['group'] == 'masters'))
            return value
        with patch.object(v, 'read', side_effect=modified):
            with self.assertRaisesRegex(ValueError, 'numerical coverage differs'):
                v.verify('long', audit_only=True)

    def test_altered_retained_timing_is_rejected(self):
        original = v.read
        def modified(path):
            value = original(path)
            if Path(path).name == 'performance-004.json':
                value = copy.deepcopy(value)
                value['pairs'][0]['cases']['native']['mean_iteration_ms'] += 1
            return value
        with patch.object(v, 'read', side_effect=modified):
            with self.assertRaisesRegex(ValueError, 'statistics differ'):
                v.verify('long')


if __name__ == '__main__':
    unittest.main()
