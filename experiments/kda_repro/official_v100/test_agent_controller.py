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
        prefill_source = Path(__file__).with_name("gdn_prefill.py").read_text()
        controller.validate_source(prefill_source)

    def test_forbidden_import_and_file_call_are_rejected(self):
        with self.assertRaises(ValueError):
            controller.validate_source("import os\ndef triton_candidate():\n    pass\n")
        with self.assertRaises(ValueError):
            controller.validate_source("def triton_candidate():\n    return open('x')\n")
        with self.assertRaisesRegex(ValueError, "alias"):
            controller.validate_source("import torch as safe\ndef triton_candidate():\n    pass\n")

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

    def test_exact_replacements_materialize_a_candidate(self):
        original = "A = 1\ndef triton_candidate():\n    return A\n"
        proposal = {"rationale": "test one bounded change",
                    "replacements": [{"old": "A = 1", "new": "A = 2"}]}
        source, rationale = controller.materialize_proposal(proposal, original)
        self.assertIn("A = 2", source)
        self.assertEqual(rationale, proposal["rationale"])

    def test_replacements_must_be_unique_and_change_source(self):
        original = "A = 1\nA = 1\ndef triton_candidate():\n    return A\n"
        with self.assertRaisesRegex(ValueError, "exactly once"):
            controller.materialize_proposal(
                {"rationale": "x", "replacements": [{"old": "A = 1", "new": "A = 2"}]},
                original)
        with self.assertRaisesRegex(ValueError, "no source change"):
            controller.materialize_proposal(
                {"rationale": "x", "replacements": [{"old": "return A", "new": "return A"}]},
                original)

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

    def test_load_best_uses_latest_candidate_status(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "candidates/seed").mkdir(parents=True)
            (workspace / "candidates/agent").mkdir(parents=True)
            (workspace / "candidates/seed/candidate.py").write_text("seed")
            (workspace / "candidates/agent/candidate.py").write_text("agent")
            records = [
                {"candidate_id": "seed", "status": "promoted", "score_ms": 2.0},
                {"candidate_id": "agent", "status": "promoted", "score_ms": 1.0},
                {"candidate_id": "agent", "status": "demoted", "score_ms": 1.0},
            ]
            (workspace / "candidates.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records))
            candidate_id, score, path = controller.load_best(workspace)
            self.assertEqual((candidate_id, score), ("seed", 2.0))
            self.assertEqual(path, workspace / "candidates/seed/candidate.py")

    def test_openai_compatible_endpoint(self):
        with patch.dict(os.environ, {"KDA_LLM_BASE_URL": "https://example.test/v1"}, clear=True):
            self.assertEqual(controller.api_endpoint(),
                             "https://example.test/v1/chat/completions")


if __name__ == "__main__":
    unittest.main()
