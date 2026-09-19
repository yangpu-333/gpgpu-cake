import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import MagicMock

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

    def test_empty_content_has_clear_error(self):
        for content in (None, "", "   "):
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "empty"):
                controller.parse_json_content(content)

    def test_empty_api_content_preserves_diagnostics(self):
        payload = {"id": "request-1", "model": "test-model", "created": 1,
                   "usage": {"completion_tokens": 8000},
                   "choices": [{"finish_reason": "length",
                                "message": {"role": "assistant", "content": None}}]}
        response = MagicMock()
        response.read.return_value = json.dumps(payload).encode()
        response.__enter__.return_value = response
        with patch.dict(os.environ, {"KDA_LLM_API_KEY": "secret",
                                     "KDA_LLM_MODEL": "test-model",
                                     "KDA_LLM_BASE_URL": "https://example.test/v1"}, clear=True), \
             patch("urllib.request.urlopen", return_value=response), \
             self.assertRaises(controller.LLMResponseError) as caught:
            controller.call_llm([{"role": "user", "content": "x"}], True,
                                max_tokens=8000)
        self.assertEqual(caught.exception.metadata["finish_reason"], "length")
        self.assertEqual(caught.exception.metadata["content_type"], "NoneType")

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
