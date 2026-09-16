"""CPU-only tests for Megatron RMSNorm route selection."""

import unittest

import megatron_rmsnorm_route_probe as probe


class MegatronRMSNormRouteProbeTests(unittest.TestCase):
    def test_wrapped_torch_norm_is_local_gpt_rmsnorm_route(self):
        route = probe.select_route("WrappedTorchNorm")
        self.assertEqual(route["kind"], "gpt_local_wrapped_torch_rmsnorm")
        self.assertTrue(route["rmsnorm_supported_by_selected_factory"])

    def test_unexpected_factory_is_not_mislabeled_as_rmsnorm(self):
        route = probe.select_route("FusedLayerNorm")
        self.assertEqual(route["kind"], "unexpected_gpt_local_rmsnorm_factory")
        self.assertFalse(route["rmsnorm_supported_by_selected_factory"])


if __name__ == "__main__":
    unittest.main()
