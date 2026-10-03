"""CPU-only metadata mocks exercise installation, guards and native fallback."""
import hashlib
import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


PATH = Path(__file__).resolve().parents[1] / "kernel_adapter.py"
SPEC = importlib.util.spec_from_file_location("tested_kernel_adapter", PATH)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
REAL_VERIFY_MEGATRON = adapter._verify_megatron


class Device:
    def __init__(self, kind="cuda", index=0):
        self.type, self.index = kind, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)

    def __str__(self):
        return self.type + (":" + str(self.index) if self.type == "cuda" else "")


class Tensor:
    def __init__(self, shape, dtype="torch.float32", device=None, contiguous=True, label="tensor"):
        self.shape = tuple(shape)
        self.dtype, self.device = dtype, device or Device()
        self.is_cuda = self.device.type == "cuda"
        self._contiguous, self.label = contiguous, label

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def is_contiguous(self):
        return self._contiguous

    def stride(self):
        result, product = [], 1
        for size in reversed(self.shape):
            result.append(product)
            product *= size
        return tuple(reversed(result))

    def transpose(self, first, second):
        shape = list(self.shape)
        shape[first], shape[second] = shape[second], shape[first]
        return Tensor(shape, self.dtype, self.device, False, self.label)

    def contiguous(self):
        return Tensor(self.shape, self.dtype, self.device, True, self.label)


class RMSNorm:
    def __init__(self, hidden):
        # Installation must work before model.cuda().
        self.weight = Tensor((hidden,), device=Device("cpu", None))
        self.eps = 1e-5
        self.native_calls = 0

    def forward(self, hidden):
        self.native_calls += 1
        return Tensor(hidden.shape, label="native_norm")


class Layer:
    def __init__(self, hidden):
        self.pre_mlp_layernorm = RMSNorm(hidden)
        self.native_calls = 0

    def self_attn_bda(self, training, fused):
        def bda(values, residual, dropout):
            self.native_calls += 1
            return Tensor(residual.shape, residual.dtype, residual.device, label="native_bda")
        return bda


class GPTModel:
    __module__ = "megatron.core.models.gpt.gpt_model"

    def __init__(self):
        self.config = types.SimpleNamespace(
            num_layers=4, hidden_size=512, num_attention_heads=8, normalization="RMSNorm",
            layernorm_epsilon=1e-5, hidden_dropout=0.0, attention_dropout=0.0,
            add_bias_linear=False, sequence_parallel=False, tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1, context_parallel_size=1, fp16=False, bf16=False,
            cross_entropy_loss_fusion=False)
        self.vocab_size = 1024
        self.training = True
        self.decoder = types.SimpleNamespace(layers=[Layer(512) for _ in range(4)])
        self.tp_group = types.SimpleNamespace(size=lambda: 1)
        self.native_loss_calls = 0

    def compute_language_model_loss(self, labels, logits):
        self.native_loss_calls += 1
        return Tensor(labels.shape, label="native_loss")


def fake_candidate():
    torch = types.SimpleNamespace(
        __version__="2.4.1", float32="torch.float32", float16="torch.float16",
        bfloat16="torch.bfloat16", int64="torch.int64", nn=types.SimpleNamespace(RMSNorm=RMSNorm),
        distributed=types.SimpleNamespace(is_initialized=lambda: True, get_world_size=lambda: 1),
        cuda=types.SimpleNamespace(get_device_name=mock.Mock(return_value="Iluvatar BI-V150")))
    module = types.SimpleNamespace(torch=torch, triton=types.SimpleNamespace(__version__="2.1.0"),
                                   _cuda_autocast_enabled=lambda: False)
    module.fused = mock.Mock(side_effect=lambda x, residual, weight, epsilon:
                             (Tensor(residual.shape, label="candidate_norm"),
                              Tensor(residual.shape, label="candidate_residual")))
    module.cross_entropy = mock.Mock(side_effect=lambda logits, target:
                                     Tensor(target.shape, label="candidate_loss"))
    return module


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.model, self.candidate = GPTModel(), fake_candidate()
        self.verify = mock.patch.object(adapter, "_verify_megatron", return_value={
            "megatron_commit": adapter.PINNED_MEGATRON,
            "native_sources_sha256": dict(adapter.NATIVE_SOURCE_SHA256)})
        self.importer = mock.patch.object(adapter, "_import_candidate", return_value=self.candidate)
        self.verify.start()
        self.importer.start()
        self.addCleanup(self.verify.stop)
        self.addCleanup(self.importer.stop)

    def install(self):
        return adapter.install_kernels(self.model, enabled=True, megatron_root="unused-test-checkout")

    def gpu_weights(self):
        for layer in self.model.decoder.layers:
            layer.pre_mlp_layernorm.weight.device = Device()
            layer.pre_mlp_layernorm.weight.is_cuda = True

    def residual(self, training=True, fused=False, dropout=0.0, bias=None, shape=(128, 2, 512), **kwargs):
        layer = self.model.decoder.layers[0]
        x, residual = Tensor(shape, **kwargs), Tensor(shape, **kwargs)
        summed = layer.self_attn_bda(training, fused)((x, bias), residual, dropout)
        return layer.pre_mlp_layernorm.forward(summed)

    def ce(self, shape=(128, 2, 1024), **kwargs):
        return self.model.compute_language_model_loss(Tensor((shape[1], shape[0]), dtype="torch.int64"),
                                                       Tensor(shape, **kwargs))

    def test_disabled_is_noop_and_does_not_read_or_import(self):
        before = dict(vars(self.model))
        with mock.patch.object(adapter, "_verify_megatron") as verify, mock.patch.object(adapter, "_load_candidate") as load:
            handle = adapter.install_kernels(self.model, enabled=False)
        self.assertEqual(vars(self.model), before)
        verify.assert_not_called()
        load.assert_not_called()
        self.assertFalse(handle.snapshot()["active"])
        self.assertFalse(handle.snapshot()["provenance_verified"])

    def test_install_allows_cpu_weights_and_is_idempotent(self):
        handle = self.install()
        self.assertIs(self.install(), handle)
        self.assertTrue(handle.active)
        self.candidate.torch.cuda.get_device_name.assert_not_called()
        self.assertEqual(handle.snapshot()["counts"]["residual"]["candidate_api_calls"], 0)

    def test_disabled_cannot_relabel_an_installed_model_as_baseline(self):
        self.install()
        with self.assertRaises(adapter.AdapterError):
            adapter.install_kernels(self.model, enabled=False)

    def test_uninstall_restores_original_bindings(self):
        model_before = dict(vars(self.model))
        layer = self.model.decoder.layers[0]
        layer_before, norm_before = dict(vars(layer)), dict(vars(layer.pre_mlp_layernorm))
        handle = self.install()
        self.assertTrue(handle.uninstall())
        self.assertFalse(handle.uninstall())
        self.assertEqual(vars(self.model), model_before)
        self.assertEqual(vars(layer), layer_before)
        self.assertEqual(vars(layer.pre_mlp_layernorm), norm_before)

    def test_conflicting_hook_does_not_get_overwritten_on_uninstall(self):
        handle = self.install()
        self.model.compute_language_model_loss = lambda labels, logits: None
        with self.assertRaises(adapter.AdapterError):
            handle.uninstall()
        self.assertTrue(handle.active)

    def test_supported_residual_consumes_exact_adjacent_result(self):
        handle = self.install()
        self.gpu_weights()
        self.assertEqual(self.residual().label, "candidate_norm")
        counts = handle.snapshot()["counts"]["residual"]
        self.assertEqual((counts["candidate_api_calls"], counts["cached_norm_calls"]), (1, 1))
        self.assertEqual(counts["native_bda_calls"], 0)
        self.assertEqual(self.model.decoder.layers[0].native_calls, 0)

    def test_pending_result_is_not_used_for_a_different_tensor(self):
        handle = self.install()
        self.gpu_weights()
        layer = self.model.decoder.layers[0]
        summed = layer.self_attn_bda(True, False)((Tensor((128, 2, 512)), None), Tensor((128, 2, 512)), 0.0)
        actual = layer.pre_mlp_layernorm.forward(Tensor(summed.shape))
        self.assertEqual(actual.label, "native_norm")
        self.assertEqual(layer.pre_mlp_layernorm.forward(summed).label, "native_norm")
        self.assertEqual(handle.snapshot()["counts"]["residual"]["cached_norm_calls"], 0)

    def test_residual_known_unsupported_calls_use_native(self):
        for kwargs, reason in (({"dropout": 0.1}, "dropout_or_bias"),
                               ({"bias": object()}, "dropout_or_bias"),
                               ({"training": False}, "not_training"),
                               ({"fused": True}, "native_bias_dropout_fusion_requested"),
                               ({"shape": (129, 2, 512)}, "unverified_shape"),
                               ({"contiguous": False}, "layout"),
                               ({"device": Device("cpu", None)}, "non_cuda_tensors")):
            with self.subTest(reason=reason):
                model = self.model
                self.model = GPTModel()
                handle = self.install()
                self.gpu_weights()
                self.assertEqual(self.residual(**kwargs).label, "native_norm")
                self.assertEqual(handle.snapshot()["counts"]["residual"]["fallback_reasons"].get(reason), 1)
                self.model = model

    def test_supported_ce_transposes_native_layout_and_returns_batch_sequence(self):
        handle = self.install()
        actual = self.ce()
        self.assertEqual(actual.shape, (2, 128))
        self.assertTrue(actual.is_contiguous())
        target = self.candidate.cross_entropy.call_args.args[1]
        self.assertEqual(target.shape, (128, 2))
        self.assertTrue(target.is_contiguous())
        counts = handle.snapshot()["counts"]["cross_entropy"]
        self.assertEqual((counts["candidate_api_calls"], counts["native_calls"]), (1, 0))

    def test_ce_outside_tested_vocabulary_uses_native(self):
        handle = self.install()
        self.assertEqual(self.ce(shape=(128, 2, 32000)).label, "native_loss")
        self.assertEqual(handle.snapshot()["counts"]["cross_entropy"]["fallback_reasons"],
                         {"unverified_shape_or_vocabulary": 1})

    def test_ce_tp_greater_than_one_uses_native(self):
        handle = self.install()
        self.model.tp_group = types.SimpleNamespace(size=lambda: 2)
        self.assertEqual(self.ce().label, "native_loss")
        self.assertEqual(handle.snapshot()["counts"]["cross_entropy"]["fallback_reasons"],
                         {"tensor_parallel_group": 1})

    def test_world_size_greater_than_one_uses_native(self):
        handle = self.install()
        self.candidate.torch.distributed.get_world_size = lambda: 2
        self.assertEqual(self.ce().label, "native_loss")
        self.assertEqual(handle.snapshot()["counts"]["cross_entropy"]["fallback_reasons"],
                         {"unverified_world_size": 1})

    def test_verified_holdout_configuration_and_shapes(self):
        self.model.config.num_layers, self.model.config.hidden_size = 2, 256
        self.model.config.num_attention_heads, self.model.vocab_size = 4, 512
        self.model.decoder.layers = [Layer(256), Layer(256)]
        handle = self.install()
        self.gpu_weights()
        self.assertEqual(self.residual(shape=(17, 2, 256)).label, "candidate_norm")
        self.assertEqual(self.ce(shape=(17, 2, 512)).label, "candidate_loss")
        self.assertEqual(handle.snapshot()["counts"]["cross_entropy"]["candidate_api_calls"], 1)

    def test_autocast_is_reported_as_compatibility_not_triton_evidence(self):
        handle = self.install()
        self.gpu_weights()
        self.candidate._cuda_autocast_enabled = lambda: True
        self.residual()
        self.ce(dtype="torch.bfloat16")
        counts = handle.snapshot()["counts"]
        self.assertEqual(counts["residual"]["autocast_compatibility_calls"], 1)
        self.assertEqual(counts["cross_entropy"]["autocast_torch_compatibility_calls"], 1)
        self.assertEqual(counts["cross_entropy"]["fp32_triton_eligible_calls"], 0)

    def test_other_gpu_architecture_uses_native(self):
        handle = self.install()
        self.candidate.torch.cuda.get_device_name.return_value = "NVIDIA H100"
        self.assertEqual(self.ce().label, "native_loss")
        self.assertEqual(handle.snapshot()["counts"]["cross_entropy"]["fallback_reasons"],
                         {"unverified_gpu_architecture": 1})

    def test_candidate_failure_propagates_without_native_retry(self):
        self.install()
        self.candidate.cross_entropy.side_effect = RuntimeError("GPU launch failed")
        with self.assertRaisesRegex(RuntimeError, "GPU launch failed"):
            self.ce()
        self.assertEqual(self.model.native_loss_calls, 0)

    def test_unsupported_model_config_is_rejected_without_partial_install(self):
        self.model.config.hidden_dropout = 0.1
        with self.assertRaisesRegex(adapter.AdapterError, "hidden_dropout"):
            self.install()
        self.assertNotIn(adapter._MARKER, vars(self.model))
        self.assertNotIn("forward", vars(self.model.decoder.layers[0].pre_mlp_layernorm))

    def test_unverified_ffn_architecture_is_rejected(self):
        self.model.config.ffn_hidden_size = 1536
        with self.assertRaisesRegex(adapter.AdapterError, "ffn_hidden_size"):
            self.install()
        self.assertNotIn(adapter._MARKER, vars(self.model))

    def test_candidate_source_hash_is_checked_before_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.py"
            path.write_text("raise AssertionError('must never execute')", encoding="utf-8")
            with mock.patch.object(adapter, "_import_candidate") as importer:
                with self.assertRaisesRegex(adapter.AdapterError, "SHA256"):
                    adapter._load_candidate(path)
                importer.assert_not_called()

    def test_native_source_modification_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".git").mkdir()
            (root / ".git/HEAD").write_text(adapter.PINNED_MEGATRON + "\n", encoding="ascii")
            relative = "megatron/core/fixture.py"
            path = root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(b"original")
            expected = {relative: hashlib.sha256(b"original").hexdigest()}
            # Invoke the real verifier while the other installation tests mock
            # its network-free provenance result.
            with mock.patch.object(adapter, "NATIVE_SOURCE_SHA256", expected):
                REAL_VERIFY_MEGATRON(root)
                path.write_bytes(b"modified")
                with self.assertRaisesRegex(adapter.AdapterError, "differs"):
                    REAL_VERIFY_MEGATRON(root)

    def test_snapshot_is_a_copy_without_runtime_tensors(self):
        handle = self.install()
        self.ce()
        snapshot = handle.snapshot()
        snapshot["counts"]["cross_entropy"]["candidate_api_calls"] = 999
        self.assertEqual(handle.snapshot()["counts"]["cross_entropy"]["candidate_api_calls"], 1)
        import json
        json.dumps(handle.snapshot())


if __name__ == "__main__":
    unittest.main()
