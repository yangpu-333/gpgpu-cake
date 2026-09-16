"""CPU-only tests for Megatron RMSNorm shape-capture bookkeeping."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


INSTRUMENTATION = Path(__file__).resolve().parent / "instrumentation"
sys.path.insert(0, str(INSTRUMENTATION))
import rmsnorm_capture as capture  # noqa: E402
sys.path.pop(0)

import summarize_rmsnorm_capture as summary  # noqa: E402


class FakeTensor:
    def __init__(self, shape=(2, 3, 5), pointer=1):
        self.shape = shape
        self.dtype = "torch.float16"
        self.device = "cuda:0"
        self.requires_grad = True
        self.is_cuda = True
        self._pointer = pointer

    def stride(self):
        stride, result = 1, []
        for dimension in reversed(self.shape):
            result.append(stride)
            stride *= dimension
        return tuple(reversed(result))

    def is_contiguous(self):
        return True

    def data_ptr(self):
        return self._pointer


class RMSNormCaptureTests(unittest.TestCase):
    def test_default_and_configured_target_names(self):
        self.assertEqual(capture.target_names_from_environment({}), capture.DEFAULT_TARGETS)
        self.assertEqual(
            capture.target_names_from_environment({capture.ENV_TARGETS: "A, B ,,"}), ("A", "B")
        )

    def test_tensor_and_tuple_contract_metadata(self):
        input_tensor = FakeTensor()
        normalized = FakeTensor(pointer=2)
        residual = FakeTensor(pointer=1)
        info = capture.tensor_metadata(input_tensor)
        self.assertEqual(info["flattened_rows"], 6)
        self.assertEqual(info["hidden_size"], 5)
        contract = capture.output_contract_metadata((normalized, residual), input_tensor)
        self.assertTrue(contract["passed"])
        self.assertTrue(contract["residual_aliases_input"])
        self.assertFalse(capture.output_contract_metadata(normalized, input_tensor)["passed"])

    def test_summary_groups_ranks_and_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = {
                "environment": {"rank": "0"},
                "records": [{"status": "captured", "module": {"class_name": "TEFusedResidualRMSNorm"},
                             "input": {"shape": [4, 2, 8], "dtype": "torch.float16", "device": "cuda:0"},
                             "sampled_forward_event_ms": 0.2}],
            }
            second = {
                "environment": {"rank": "1"},
                "records": [{"status": "captured", "module": {"class_name": "TEFusedResidualRMSNorm"},
                             "input": {"shape": [4, 2, 8], "dtype": "torch.float16", "device": "cuda:0"},
                             "sampled_forward_event_ms": 0.4}],
            }
            (root / "rmsnorm-capture-a.json").write_text(json.dumps(first), encoding="utf-8")
            (root / "rmsnorm-capture-b.json").write_text(json.dumps(second), encoding="utf-8")
            report = summary.summarize_reports(root)
            self.assertEqual(report["status"], "target_calls_captured")
            self.assertEqual(len(report["workloads"]), 1)
            self.assertEqual(report["workloads"][0]["observed_calls"], 2)
            self.assertEqual(report["workloads"][0]["ranks"], ["0", "1"])
            self.assertAlmostEqual(report["workloads"][0]["sampled_event_ms"]["median"], 0.3)

    def test_summary_distinguishes_no_target_calls_from_missing_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "rmsnorm-capture-empty.json").write_text(
                json.dumps({"environment": {"rank": "0"}, "records": []}), encoding="utf-8"
            )
            self.assertEqual(summary.summarize_reports(root)["status"], "no_target_module_calls")
            empty = Path(folder) / "empty"
            empty.mkdir()
            self.assertEqual(summary.summarize_reports(empty)["status"], "no_capture_reports_found")


if __name__ == "__main__":
    unittest.main()
