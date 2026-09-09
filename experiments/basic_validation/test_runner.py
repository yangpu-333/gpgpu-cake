"""Tests of functional bookkeeping; GPU kernels require real-device validation."""

import contextlib
import io
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

import run


class RunnerTests(unittest.TestCase):
    def test_tiled_matmul_handles_partial_tiles(self):
        rng = random.Random(41)
        for m, n, k in ((1, 1, 1), (2, 7, 3), (9, 5, 11)):
            a = [[rng.uniform(-2, 2) for _ in range(k)] for _ in range(m)]
            b = [[rng.uniform(-2, 2) for _ in range(n)] for _ in range(k)]
            expected = run.flatten(run.reference_matmul(a, b))
            for tile in (1, 2, 4, 16):
                with self.subTest(shape=(m, n, k), tile=tile):
                    actual = run.flatten(run.tiled_matmul(a, b, tile))
                    self.assertTrue(run.compare_values(actual, expected)["passed"])

    def test_tiled_matmul_rejects_invalid_tile(self):
        with self.assertRaises(ValueError):
            run.tiled_matmul([[1]], [[2]], 0)

    def test_nonfinite_and_length_mismatch_are_not_correct(self):
        for values in ([math.nan], [math.inf], []):
            with self.subTest(values=values):
                self.assertFalse(run.compare_values(values, [1.0])["passed"])

    def test_comparison_uses_elementwise_tolerance(self):
        self.assertFalse(run.compare_values([0.01], [0], atol=0.001, rtol=0.1)["passed"])
        self.assertTrue(run.compare_values([100.01], [100], atol=0.001, rtol=0.001)["passed"])

    def test_failed_or_untimed_candidate_cannot_win(self):
        def item(kind, passed, ms):
            return {"kind": kind, "correctness": {"passed": passed}, "median_ms": ms}
        valid = item("candidate", True, 4.0)
        records = [item("candidate", False, 0.01), item("baseline", True, 0.1),
                   item("candidate", True, math.nan), item("candidate", True, -1),
                   {"kind": "candidate", "correctness": {"passed": True}}, valid]
        self.assertIs(run.choose_fastest(records), valid)
        self.assertIsNone(run.choose_fastest(records[:-1]))

    def test_cpu_report_does_not_claim_gpu_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "cpu.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run.main(["--mode", "cpu", "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(report["status"], "cpu_check_passed")
            self.assertEqual(len(report["records"]), 18)
            self.assertNotIn("selections", report)
            self.assertTrue(all("median_ms" not in row for row in report["records"]))

    def test_cli_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "previous.json"
            output.write_text("original", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                run.main(["--mode", "cpu", "--output", str(output)])
            self.assertEqual(exc.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
