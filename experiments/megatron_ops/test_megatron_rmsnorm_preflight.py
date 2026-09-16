"""CPU-only tests for preflight validation rules."""

import unittest

import megatron_rmsnorm_preflight as preflight


class MegatronRMSNormPreflightTests(unittest.TestCase):
    def test_dimensions_describe_the_tensor_contract(self):
        result = preflight.validate_dimensions(8, 2, 1024, 8)
        self.assertEqual(result["shape"], [8, 2, 1024])
        self.assertEqual(result["flattened_rows"], 16)
        with self.assertRaises(ValueError):
            preflight.validate_dimensions(8, 2, 1025, 8)

    def test_status_requires_the_expected_class_and_both_passes(self):
        self.assertEqual(
            preflight.smoke_status(preflight.EXPECTED_CLASS, True, True), "module_smoke_passed"
        )
        self.assertEqual(
            preflight.smoke_status(preflight.EXPECTED_CLASS, True, False), "module_smoke_failed"
        )
        self.assertEqual(preflight.smoke_status("RMSNorm", True, True), "fused_module_not_selected")


if __name__ == "__main__":
    unittest.main()
