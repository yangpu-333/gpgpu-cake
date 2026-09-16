"""CPU-only tests for Megatron RMSNorm route selection."""

import unittest

import megatron_rmsnorm_route_probe as probe


class MegatronRMSNormRouteProbeTests(unittest.TestCase):
    def test_transformer_engine_precedes_other_backends(self):
        route = probe.select_route(True, True, "FusedLayerNorm")
        self.assertEqual(route["kind"], "transformer_engine")
        self.assertTrue(route["rmsnorm_supported_by_selected_factory"])

    def test_apex_factory_does_not_claim_rmsnorm_support(self):
        route = probe.select_route(False, True, "FusedLayerNorm")
        self.assertEqual(route["kind"], "apex_fused_layernorm")
        self.assertFalse(route["rmsnorm_supported_by_selected_factory"])

    def test_wrapped_torch_norm_is_native_rmsnorm_route(self):
        route = probe.select_route(False, False, "WrappedTorchNorm")
        self.assertEqual(route["kind"], "wrapped_torch_rmsnorm")
        self.assertTrue(route["rmsnorm_supported_by_selected_factory"])


if __name__ == "__main__":
    unittest.main()
