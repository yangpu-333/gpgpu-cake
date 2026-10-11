"""Reject loss of numerical coverage, altered timing and resource-gate bypass."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
import verify_completion_retry01 as completion
import verify_receipts_retry01_mount as receipts


class FreshEvidenceGates(unittest.TestCase):
    def test_missing_master_tensor_is_rejected(self):
        original = receipts.read
        def changed(path):
            result = original(path)
            if Path(path).name == 'comparison-control-004.json':
                result = copy.deepcopy(result)
                rows = result['tensor_comparisons']
                rows.pop(next(i for i, row in enumerate(rows) if row['group'] == 'masters'))
            return result
        with patch.object(receipts, 'read', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'numerical coverage differs'):
                receipts.verify('long', audit_only=True)

    def test_altered_retained_timing_is_rejected(self):
        original = receipts.read
        def changed(path):
            result = original(path)
            if Path(path).name == 'performance-004.json':
                result = copy.deepcopy(result)
                result['pairs'][0]['cases']['native']['mean_iteration_ms'] += 1
            return result
        with patch.object(receipts, 'read', side_effect=changed):
            with self.assertRaisesRegex(ValueError, 'statistics differ'):
                receipts.verify('long', confirmation=True)

    def test_busy_post_run_resources_block_acceptance(self):
        original = completion.read
        def changed(path):
            result = original(path)
            if Path(path).name == 'browser-formal-long-retry01-resource_confirmation-02.json':
                result = copy.deepcopy(result)
                result['occupancy_after']['passed'] = False
            return result
        with patch.object(completion, 'read', side_effect=changed):
            with self.assertRaises(AssertionError):
                completion.verify(long_only=True)

    def test_diagnostic_receipt_cannot_be_relabelled_passed(self):
        original = completion.read
        def changed(path):
            result = original(path)
            if Path(path).name == 'browser-formal-long-retry01-primary-01.json':
                result = copy.deepcopy(result)
                result['status'] = 'passed'
            return result
        with patch.object(completion, 'read', side_effect=changed):
            with self.assertRaises(AssertionError):
                completion.verify(long_only=True)


if __name__ == '__main__':
    unittest.main()
