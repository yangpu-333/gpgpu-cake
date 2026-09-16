"""Exercise raw API failure reporting with a fake library, never a real GPU."""

import contextlib
import ctypes
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import stream_api_diagnostics as probe


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class FakeLibrary:
    def __init__(self, failures=None, corrupt_copy=False):
        self.failures = failures or {}
        self.corrupt_copy = corrupt_copy
        self.calls = []
        self.value = 0

    def __getattr__(self, name):
        def invoke(*args):
            self.calls.append(name)
            if name == "cudaGetErrorString":
                return b"fake runtime error"
            if name == "cudaGetErrorName":
                return b"cudaErrorNotSupported"
            code = self.failures.get(name, 0)
            if code:
                return code
            if name == "cudaGetDeviceCount":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 2
            elif name == "cudaDeviceGetStreamPriorityRange":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_int))[0] = 0
                ctypes.cast(args[1], ctypes.POINTER(ctypes.c_int))[0] = -1
            elif name == "cudaMalloc":
                ctypes.cast(args[0], ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
            elif name == "cudaMemcpy":
                kind = getattr(args[3], "value", args[3])
                if kind == 1:
                    self.value = ctypes.cast(args[1], ctypes.POINTER(ctypes.c_uint32))[0]
                elif kind == 2:
                    ctypes.cast(args[0], ctypes.POINTER(ctypes.c_uint32))[0] = self.value ^ int(self.corrupt_copy)
                else:
                    raise AssertionError(f"unexpected memcpy kind: {kind}")
            return 0
        function = FakeFunction(invoke)
        setattr(self, name, function)
        return function


class StreamAPITests(unittest.TestCase):
    def run_case(self, case, library):
        with patch("ctypes.CDLL", return_value=library), contextlib.redirect_stdout(io.StringIO()):
            return probe.execute_case(case, 0, "/fake/libcudart.so")

    def test_priority_failure_does_not_claim_malloc_failed(self):
        library = FakeLibrary({"cudaDeviceGetStreamPriorityRange": 801})
        result = self.run_case("raw_priority_range", library)
        self.assertFalse(result["passed"])
        self.assertIn("cudaDeviceGetStreamPriorityRange", result["active_stage"])
        self.assertNotIn("cudaMalloc", library.calls)

    def test_raw_copy_requires_correct_values_and_frees_allocation(self):
        for corrupt in (False, True):
            with self.subTest(corrupt=corrupt):
                library = FakeLibrary(corrupt_copy=corrupt)
                result = self.run_case("raw_malloc_copy", library)
                self.assertEqual(result["passed"], not corrupt)
                self.assertEqual(library.calls.count("cudaFree"), 1)

    def test_copy_failure_preserves_stage_despite_cleanup(self):
        library = FakeLibrary({"cudaMemcpy": 801})
        result = self.run_case("raw_malloc_copy", library)
        self.assertFalse(result["passed"])
        self.assertIn("cudaMemcpy", result["active_stage"])
        self.assertEqual(library.calls.count("cudaFree"), 1)

    def test_free_failure_cannot_pass(self):
        result = self.run_case("raw_malloc_copy", FakeLibrary({"cudaFree": 801}))
        self.assertFalse(result["passed"])

    def test_missing_control_node_is_reported_without_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            node = Path(directory) / "itrctl"
            result = probe.device_info(node)
            self.assertFalse(result["exists"])
            self.assertFalse(node.exists())

    def test_path_variant_preserves_visibility_and_original_environment(self):
        before = {"IX_VISIBLE_DEVICES": "platform-gpu-id", "CUDA_VISIBLE_DEVICES": "0",
                  "LD_PRELOAD": "/platform/hook.so", "LD_LIBRARY_PATH": "/original/lib"}
        environment = before.copy()
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "libcuda.so.1").write_text("fake", encoding="utf-8")
            variants, info = probe.build_variants(environment, Path(directory))
        self.assertEqual(environment, before)
        self.assertEqual(len(variants), 2)
        for name, settings in variants:
            for key in ("IX_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "LD_PRELOAD"):
                self.assertEqual(settings[key], before[key])


if __name__ == "__main__":
    unittest.main()
