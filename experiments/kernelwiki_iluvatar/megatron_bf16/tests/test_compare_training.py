"""CPU-only regression fixtures for BF16 audit completeness and rounding risks."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "compare_training.py"
SPEC = importlib.util.spec_from_file_location("bf16_compare_training", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.native, self.optimized = self.root / "native", self.root / "optimized"
        for case in (self.native, self.optimized):
            (case / "snapshots").mkdir(parents=True)
        self.comparison = MODULE.Comparison(self.native, self.optimized, torch)
        self.comparison.model = {"parameters": 3, "micro_batch_size": 2, "sequence_length": 4}
        self.comparison.parameter_names = {arm: {"weight", "bias"} for arm in ("native", "optimized")}

    @staticmethod
    def state():
        # Native BF16 Megatron constructs linear/embedding weights in BF16,
        # while normalization parameters begin as FP32 before Float16Module.
        return {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
                "bias": torch.tensor([0.5], dtype=torch.float32), "layer._extra_state": None}

    def parameters(self, dtype):
        return {name: value.to(dtype).clone() for name, value in self.state().items()
                if torch.is_tensor(value)}

    def write_case(self, case, *, master_delta=0.0, omit_master=None):
        snapshots = case / "snapshots"
        torch.save(self.state(), snapshots / "model0-initial.pt")
        model = self.parameters(torch.bfloat16)
        masters = {name: value.float() for name, value in model.items()}
        torch.save({"model_parameters": model, "master_parameters": masters}, snapshots / "actual-start.pt")
        for step in range(1, 4):
            torch.save({"output": torch.full((2, 4), 2.0, dtype=torch.float32),
                        "positional_inputs": {"0": torch.arange(8).reshape(2, 4)},
                        "keyword_inputs": {"loss_mask": torch.ones((2, 4))}},
                       snapshots / ("model0-forward-%03d.pt" % step))
            for label, values in (("params", model),
                                  ("wgrads", {name: torch.zeros_like(value, dtype=torch.float32)
                                              for name, value in model.items()})):
                directory = snapshots / label / ("iter_%07d" % step)
                directory.mkdir(parents=True)
                torch.save({"model_chunk0": values}, directory / "mp_rank_00.pth")
            after = {name: value.clone() for name, value in masters.items()}
            if master_delta:
                after["weight"][0] += master_delta
            if omit_master:
                del after[omit_master]
            torch.save(after, snapshots / ("master-after-%03d.pt" % step))

    def run_fixture(self):
        # Source and launcher metadata have independent guards; these fixtures
        # isolate tensor audit mechanics rather than forge a real launch.
        self.comparison.metadata = lambda: None
        return self.comparison.run()

    def test_complete_small_bf16_audit_passes(self):
        self.write_case(self.native)
        self.write_case(self.optimized)
        report = self.run_fixture()
        self.assertTrue(report["passed"], [item for item in report["checks"] if not item["passed"]])
        self.assertEqual(report["tensor_groups"]["masters"], 6)
        self.assertEqual(report["tensor_groups"]["derived_loss"], 3)
        self.assertEqual(report["pre_wrap_initial_dtype_counts"],
                         {arm: {"torch.bfloat16": 1, "torch.float32": 1}
                          for arm in ("native", "optimized")})

    def test_fp16_prewrap_parameters_are_rejected(self):
        self.write_case(self.native)
        self.write_case(self.optimized)
        for case in (self.native, self.optimized):
            state = self.state()
            state["weight"] = state["weight"].half()
            torch.save(state, case / "snapshots" / "model0-initial.pt")
        report = self.run_fixture()
        self.assertFalse(report["passed"])
        self.assertTrue(any(not value["passed"] and "native BF16/FP32 construction" in value["name"]
                            for value in report["checks"]))

    def test_master_difference_hidden_by_bf16_rounding_fails(self):
        # 0.002 exceeds the fixed 0.0013 FP32 tolerance around 1.0, while
        # both values round to the same BF16 parameter representation.
        self.assertEqual(torch.tensor(1.0).bfloat16().item(), torch.tensor(1.002).bfloat16().item())
        self.write_case(self.native)
        self.write_case(self.optimized, master_delta=0.002)
        report = self.run_fixture()
        self.assertFalse(report["passed"])
        self.assertTrue(any(not value["passed"] and value["group"] == "masters"
                            for value in report["tensor_comparisons"]))
        self.assertTrue(all(value["passed"] for value in report["tensor_comparisons"]
                            if value["group"] == "params"))

    def test_missing_master_name_fails_even_if_both_arms_omit_it(self):
        self.write_case(self.native, omit_master="bias")
        self.write_case(self.optimized, omit_master="bias")
        report = self.run_fixture()
        self.assertFalse(report["passed"])
        self.assertTrue(any(not value["passed"] and "covers every recorded parameter name" in value["name"]
                            for value in report["checks"]))

    def test_bf16_master_is_rejected(self):
        values = {arm: self.parameters(torch.bfloat16) for arm in ("native", "optimized")}
        self.comparison.parameter_mapping(values, "masters", torch.float32, None)
        self.assertTrue(any(not value["passed"] and "dtype" in value["name"]
                            for value in self.comparison.checks))

    def test_bf16_gradient_is_rejected(self):
        values = {"model_chunk0": self.parameters(torch.bfloat16)}
        self.comparison.native_parameter_dump(values, values, {arm: self.state() for arm in ("native", "optimized")},
                                              None, 1, "wgrads")
        self.assertTrue(any(not value["passed"] and "dtype" in value["name"]
                            for value in self.comparison.checks))

    def test_nan_optimizer_master_is_rejected(self):
        value = torch.tensor([float("nan")], dtype=torch.float32)
        self.comparison.compare_tensor(value, value, "master", "masters")
        self.assertFalse(self.comparison.tensors[-1]["passed"])
        self.assertFalse(self.comparison.tensors[-1]["all_elements_finite"])

    def test_missing_mask_prevents_inferred_training_loss(self):
        self.comparison.derived_loss({"output": torch.ones(2, 4)}, {"output": torch.ones(2, 4)}, 1)
        self.assertTrue(any(not value["passed"] for value in self.comparison.checks))

    def test_wrong_shape_is_rejected(self):
        self.comparison.compare_tensor(torch.ones(2), torch.ones(3), "weight", "masters")
        self.assertFalse(self.comparison.tensors[-1]["passed"])

    def test_native_bf16_model_matches_rounded_master_each_step(self):
        self.write_case(self.native)
        self.write_case(self.optimized)
        path = self.native / "snapshots" / "master-after-002.pt"
        masters = torch.load(path, weights_only=True)
        masters["weight"][0] = 1.1
        torch.save(masters, path)
        report = self.run_fixture()
        self.assertFalse(report["passed"])
        self.assertTrue(any(not value["passed"] and value["group"] == "master_model_consistency"
                            for value in report["tensor_comparisons"]))


if __name__ == "__main__":
    unittest.main()
