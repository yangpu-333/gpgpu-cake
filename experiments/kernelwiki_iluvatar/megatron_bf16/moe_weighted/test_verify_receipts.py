"""Meaningful archive corruption gates: incomplete FP32 masters and altered timing."""
import unittest
from unittest.mock import patch
import verify_receipts as v

class ArchiveGates(unittest.TestCase):
    def test_missing_master_tensor_is_rejected(self):
        original=v.read
        def corrupt(path):
            value=original(path)
            if path.name=='comparison-native-003.json':
                target=next(i for i,r in enumerate(value['tensor_comparisons']) if r['group']=='masters')
                value['tensor_comparisons'].pop(target)
            return value
        with patch.object(v,'read',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'numerical coverage differs'):
                v.verify('003',v.HERE/'evidence')

    def test_altered_retained_timing_is_rejected(self):
        original=v.read
        def corrupt(path):
            value=original(path)
            if path.name=='performance-003.json':
                value['pairs'][0]['cases']['native']['mean_iteration_ms']+=1
            return value
        with patch.object(v,'read',side_effect=corrupt):
            with self.assertRaisesRegex(ValueError,'statistics differ'):
                v.verify('003',v.HERE/'evidence')

if __name__=='__main__':unittest.main()
