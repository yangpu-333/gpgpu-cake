"""Unit tests for the static parts of the Residual Add RMSNorm workflow."""

import math
import unittest

import residual_rmsnorm_autotune as runner


class ResidualRMSNormAutotuneTests(unittest.TestCase):
    def test_normalized_ir_is_stable_and_preserves_observable_residual(self):
        first = runner.normalized_ir(7, 33, "float16", 1e-6)
        second = runner.normalized_ir(7, 33, "float16", 1e-6)
        self.assertEqual(first["workload_hash"], second["workload_hash"])
        self.assertEqual(first["canonical_ir"]["domain"], {"rows": 7, "hidden": 33})
        self.assertTrue(first["canonical_ir"]["semantics"]["residual_out_is_observable"])
        self.assertEqual(first["canonical_ir"]["semantics"]["reduction_reassociation"], "not_assumed_legal")

    def test_poly_candidates_are_legal_and_cover_the_reduction(self):
        candidates = runner.generate_poly_candidates(8, 769)
        self.assertGreaterEqual(len(candidates), 3)
        for candidate in candidates:
            with self.subTest(candidate=candidate["config"]):
                self.assertTrue(candidate["static_legality"]["passed"])
                self.assertGreaterEqual(candidate["config"]["reduction_tile"], 769)
                self.assertEqual(candidate["config"]["row_tile"], 1)

    def test_illegal_schedule_and_unsupported_reduction_are_rejected(self):
        legal, reason = runner.candidate_is_legal(
            {"row_tile": 2, "reduction_tile": 1024, "vector_width": 1, "num_warps": 4}, 8, 768)
        self.assertFalse(legal)
        self.assertEqual(reason, "only_one_row_per_program_is_implemented")
        rejected = runner.generate_poly_candidates(4, runner.MAX_REDUCTION_BLOCK + 1)
        self.assertTrue(all(not item["static_legality"]["passed"] for item in rejected))

    def test_choose_fastest_requires_forward_and_backward_correctness(self):
        def record(ms, forward=True, backward=True):
            return {"kind": "candidate", "median_ms": ms,
                    "forward_correctness": {"passed": forward},
                    "backward_correctness": {"passed": backward}}
        valid = record(2.0)
        result = runner.choose_fastest([record(0.1, backward=False), record(math.nan), valid, record(3.0)])
        self.assertIs(result, valid)

    def test_shape_parser_rejects_ambiguous_inputs(self):
        self.assertEqual(runner.parse_shapes("1x2,3x4"), [(1, 2), (3, 4)])
        with self.assertRaises(Exception):
            runner.parse_shapes("1x2x3")


if __name__ == "__main__":
    unittest.main()
