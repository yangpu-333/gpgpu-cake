"""Process/result integrity tests; no GPU capability is implied by these tests."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import device_diagnostics
import runtime_diagnostics as runtime


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_variants_preserve_platform_mapping_and_parent(self):
        original = {"IX_VISIBLE_DEVICES": "GPU-platform-token", "CUDA_VISIBLE_DEVICES": "1",
                    "LD_PRELOAD": "/platform/libhook.so", "LD_LIBRARY_PATH": "/platform/lib",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"}
        before = original.copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lib").mkdir()
            variants = dict(runtime.build_variants(original, [root]))
        self.assertEqual(original, before)
        for settings in variants.values():
            for key in ("IX_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
                self.assertEqual(settings[key], before[key])
        self.assertEqual(variants["baseline"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
        self.assertNotIn("PYTORCH_NO_CUDA_MEMORY_CACHING", variants["native"])
        self.assertEqual(variants["native_no_cache"]["PYTORCH_NO_CUDA_MEMORY_CACHING"], "1")
        self.assertTrue(variants["corex0_path"]["LD_LIBRARY_PATH"].endswith("/platform/lib"))

    def test_crashed_or_timed_out_child_cannot_pass(self):
        stdout = runtime.MARKER + json.dumps({"passed": True, "active_stage": "completed"})
        for code, timed_out in ((1, False), (-11, False), (None, True)):
            result = runtime.parse_smoke({"stdout": stdout, "stderr": "", "returncode": code,
                                          "timed_out": timed_out})
            self.assertFalse(result["passed"])
        result = runtime.parse_smoke({"stdout": stdout, "stderr": "", "returncode": 0,
                                      "timed_out": False})
        self.assertTrue(result["passed"])

    def test_timeout_preserves_last_stage_and_serializes_partial_bytes(self):
        progress = runtime.STAGE_MARKER + json.dumps({"active_stage": "allocate_one_float"})
        timeout = subprocess.TimeoutExpired("child", 1, output=(progress + "\n").encode(), stderr=b"error")
        process = {"stdout": runtime.as_text(timeout.stdout), "stderr": runtime.as_text(timeout.stderr),
                   "returncode": None, "timed_out": True}
        result = runtime.parse_smoke(process)
        self.assertFalse(result["passed"])
        self.assertEqual(result["active_stage"], "allocate_one_float")
        json.dumps(result)
        with patch.object(device_diagnostics.subprocess, "run", side_effect=timeout):
            old_result = device_diagnostics.run_child("allocate", 0)
        self.assertFalse(old_result["passed"])
        json.dumps(old_result)

    def test_followup_rejects_crashed_child_and_handles_malformed_output(self):
        success = device_diagnostics.MARKER + json.dumps({"case": "allocate", "passed": True})
        crashed = subprocess.CompletedProcess("child", 1, stdout=success, stderr="crashed")
        with patch.object(device_diagnostics.subprocess, "run", return_value=crashed):
            self.assertFalse(device_diagnostics.run_child("allocate", 0)["passed"])
        malformed = subprocess.CompletedProcess("child", 0, stdout=device_diagnostics.MARKER + "{", stderr="")
        with patch.object(device_diagnostics.subprocess, "run", return_value=malformed):
            self.assertFalse(device_diagnostics.run_child("allocate", 0)["passed"])

    def test_real_child_timeout_returns_output(self):
        process = runtime.run_command([sys.executable, "-c",
                                       "import time; print('started', flush=True); time.sleep(20)"],
                                      dict(runtime.os.environ), 2)
        self.assertTrue(process["timed_out"])
        self.assertIn("started", process["stdout"])

    def test_existing_report_is_preserved_before_running_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.write_text("existing evidence", encoding="utf-8")
            with patch.object(runtime, "run_command") as run, contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runtime.main(["--output", str(output)])
            run.assert_not_called()
            self.assertEqual(output.read_text(encoding="utf-8"), "existing evidence")


if __name__ == "__main__":
    unittest.main()
