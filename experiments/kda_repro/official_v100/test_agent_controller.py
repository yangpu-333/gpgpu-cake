import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent_controller as controller


class AgentControllerTests(unittest.TestCase):
    def test_reviewed_seed_passes_static_policy(self):
        source = Path(__file__).with_name("gdn_decode.py").read_text()
        controller.validate_source(source)

    def test_forbidden_import_and_file_call_are_rejected(self):
        with self.assertRaises(ValueError):
            controller.validate_source("import os\ndef triton_candidate():\n    pass\n")
        with self.assertRaises(ValueError):
            controller.validate_source("def triton_candidate():\n    return open('x')\n")

    def test_json_fence_is_accepted(self):
        value = controller.parse_json_content('```json\n{"rationale":"x","source":"y"}\n```')
        self.assertEqual(value["source"], "y")

    def test_candidate_score_requires_complete_correct_report(self):
        workloads = []
        for index in range(2):
            workloads.append({"uuid": str(index), "candidates": {
                "candidate": {"status": "correct", "median_ms": index + 1}}})
        score = controller.candidate_score({"status": "complete", "workloads": workloads}, 2)
        self.assertAlmostEqual(score, 2 ** 0.5)
        workloads[0]["candidates"]["candidate"]["status"] = "rejected"
        with self.assertRaises(ValueError):
            controller.candidate_score({"status": "complete", "workloads": workloads}, 2)

    def test_openai_compatible_endpoint(self):
        with patch.dict(os.environ, {"KDA_LLM_BASE_URL": "https://example.test/v1"}, clear=True):
            self.assertEqual(controller.api_endpoint(),
                             "https://example.test/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
