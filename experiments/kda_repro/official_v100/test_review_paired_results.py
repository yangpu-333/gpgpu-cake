import unittest
from pathlib import Path

import review_paired_results as review


def report(candidate_ms: float, baseline_ms: float, uuid: str = "w") -> dict:
    return {
        "status": "complete",
        "baseline_source_sha256": "baseline-hash",
        "source_sha256": "candidate-hash",
        "workloads": [{
            "uuid": uuid,
            "candidates": {
                "baseline": {"status": "correct", "median_ms": baseline_ms},
                "candidate": {"status": "correct", "median_ms": candidate_ms},
            },
        }],
    }


class PairedReviewTests(unittest.TestCase):
    def test_three_reports_must_meet_threshold(self):
        reports = [(Path(f"run-{index}.json"), report(0.9, 1.0)) for index in range(3)]
        result = review.review_reports(reports, 0.01)
        self.assertTrue(result["meets_threshold"])
        self.assertAlmostEqual(result["paired_speedup"], 1 / 0.9)
        self.assertEqual(result["candidate_wins"], 3)

    def test_small_improvement_is_not_promoted(self):
        reports = [(Path(f"run-{index}.json"), report(0.995, 1.0)) for index in range(3)]
        result = review.review_reports(reports, 0.01)
        self.assertFalse(result["meets_threshold"])

    def test_mismatched_workloads_are_rejected(self):
        reports = [(Path("a"), report(0.9, 1.0, "a")),
                   (Path("b"), report(0.9, 1.0, "b")),
                   (Path("c"), report(0.9, 1.0, "a"))]
        with self.assertRaisesRegex(ValueError, "same"):
            review.review_reports(reports, 0.01)


if __name__ == "__main__":
    unittest.main()
