"""The CPU admission must reject build-source/binary tampering and wrong phase."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import verify_receipts


class RuntimeExtraTests(unittest.TestCase):
    def test_original_before_after_pair_and_tampering(self):
        original = json.loads((Path(__file__).parent / 'evidence/runtime-extra-before-perf.json').read_text())
        after = copy.deepcopy(original)
        after['phase'] = 'after-performance'
        with tempfile.TemporaryDirectory() as name:
            base = Path(name)
            (base / 'runtime-extra-before-perf.json').write_text(json.dumps(original))
            output = base / 'runtime-extra-after-perf.json'
            output.write_text(json.dumps(after))
            self.assertTrue(verify_receipts.runtime_extra(base)['reported_bytes_unchanged'])
            for path in original['files']:
                with self.subTest(file=path):
                    mutated = copy.deepcopy(after)
                    mutated['files'][path]['sha256'] = '0' * 64
                    output.write_text(json.dumps(mutated))
                    with self.assertRaises(ValueError):
                        verify_receipts.runtime_extra(base)
            for key, value in (('phase', 'before-performance'), ('megatron_commit', '0' * 40)):
                with self.subTest(key=key):
                    mutated = copy.deepcopy(after)
                    mutated[key] = value
                    output.write_text(json.dumps(mutated))
                    with self.assertRaises(ValueError):
                        verify_receipts.runtime_extra(base)


if __name__ == '__main__':
    unittest.main()
