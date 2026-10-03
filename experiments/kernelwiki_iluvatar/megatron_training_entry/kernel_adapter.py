"""Opt-in instance adapter for the pinned BI-V150 Megatron experiment.

Install on the unwrapped MCore GPTModel after construction; CPU parameters at
installation are allowed. Runtime guards determine whether each call can use
the immutable candidate. Neither Megatron files nor the parent verifier are
modified. The counters describe API dispatch, not observed GPU kernel launches.
"""
from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys
import types


PINNED_MEGATRON = "5be9626709af2722333bf54797c954c09edeada3"
CANDIDATE_SHA256 = "4a5c2180114396412b1c99907837ab970b2a9efdffe85e0a46f5527cc0790201"
NATIVE_SOURCE_SHA256 = {
    "megatron/core/transformer/torch_norm.py": "98259df6a9f8ad565160731e322c7b7b4f9b222fac2e459cd3610c0cbdad34dc",
    "megatron/core/fusions/fused_bias_dropout.py": "897e60739d918ec917f7a8bfc578ac82a5a413f3bb80314a747a18d69c7b95df",
    "megatron/core/tensor_parallel/cross_entropy.py": "136feeeb3234d3d9d0e8b4fb91d82e88d4cb951e50f2a4d85ee6a5e0974b7f5d",
    "megatron/core/models/common/language_module/language_module.py": "8426efb81706bf6669bce97d35a582ea40d13d33b6c491d9cf8af29b2861baaf",
    "megatron/core/models/gpt/gpt_model.py": "f42663bf44c39273ae4febfac9ce9e567c5d4b31bf60aa3c6df99372eff45cc1",
}
DEFAULT_CANDIDATE = (Path(__file__).resolve().parent.parent / "megatron_cc" /
                     "higher_gain/candidates/0006/candidate.py")
_MARKER = "_kernelwiki_iluvatar_adapter"
_MODULES = {}
_CONFIG_FIELDS = ("num_layers", "hidden_size", "num_attention_heads", "normalization",
                  "ffn_hidden_size", "num_query_groups", "kv_channels",
                  "layernorm_epsilon", "hidden_dropout", "attention_dropout", "add_bias_linear",
                  "sequence_parallel", "tensor_model_parallel_size", "pipeline_model_parallel_size",
                  "context_parallel_size", "fp16", "bf16")
_MODEL_CONFIGS = {(4, 512, 8, 1024), (2, 256, 4, 512)}
_RESIDUAL_SHAPES = {(256, 512), (34, 256)}
_CE_SHAPES = {(128, 2, 1024), (17, 2, 512)}


class AdapterError(RuntimeError):
    """An opt-in installation cannot establish its provenance or supported scope."""


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_megatron(root):
    root = Path(root).resolve()
    try:
        head = (root / ".git/HEAD").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise AdapterError("Cannot verify the fixed Megatron detached HEAD") from exc
    if head != PINNED_MEGATRON:
        raise AdapterError("Megatron must have the recorded detached HEAD " + PINNED_MEGATRON)
    observed = {}
    for relative, expected in NATIVE_SOURCE_SHA256.items():
        try:
            observed[relative] = _sha(root / relative)
        except OSError as exc:
            raise AdapterError("Cannot verify native Megatron source: " + relative) from exc
        if observed[relative] != expected:
            raise AdapterError("Native Megatron source differs from the verified snapshot: " + relative)
    return {"megatron_commit": head, "native_sources_sha256": observed}


def _import_candidate(path):
    name = "_kernelwiki_iluvatar_candidate_" + CANDIDATE_SHA256[:16]
    key = str(path.resolve())
    if key not in _MODULES:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
        _MODULES[key] = module
    return _MODULES[key]


def _load_candidate(path):
    path = Path(path).resolve()
    try:
        actual = _sha(path)
    except OSError as exc:
        raise AdapterError("Cannot read the immutable candidate") from exc
    if actual != CANDIDATE_SHA256:
        raise AdapterError("Candidate source SHA256 differs from approved 0006")
    candidate = _import_candidate(path)
    if not callable(getattr(candidate, "fused", None)) or not callable(getattr(candidate, "cross_entropy", None)):
        raise AdapterError("The verified candidate API is unavailable")
    if str(candidate.torch.__version__) != "2.4.1" or str(candidate.triton.__version__) != "2.1.0":
        raise AdapterError("Candidate requires the recorded Torch 2.4.1 / CoreX Triton 2.1.0 build")
    return candidate


def _model_reason(model, torch):
    model_type = type(model)
    if (model_type.__name__ != "GPTModel" or
            model_type.__module__ != "megatron.core.models.gpt.gpt_model"):
        return "unwrapped MCore GPTModel required"
    config = getattr(model, "config", None)
    if config is None:
        return "missing TransformerConfig"
    signature = (getattr(config, "num_layers", None), getattr(config, "hidden_size", None),
                 getattr(config, "num_attention_heads", None), getattr(model, "vocab_size", None))
    if signature not in _MODEL_CONFIGS:
        return "model configuration is outside the two verified small GPT configurations"
    for field, expected in (("ffn_hidden_size", 4 * config.hidden_size),
                            ("num_query_groups", config.num_attention_heads),
                            ("kv_channels", config.hidden_size // config.num_attention_heads)):
        if getattr(config, field, expected) != expected:
            return "unsupported model setting: " + field
    activation = getattr(config, "activation_func", None)
    if activation is not None and getattr(activation, "__name__", None) != "gelu":
        return "unverified activation function"
    for field, expected in (("normalization", "RMSNorm"), ("layernorm_epsilon", 1e-5),
                            ("hidden_dropout", 0.0), ("attention_dropout", 0.0),
                            ("add_bias_linear", False), ("sequence_parallel", False),
                            ("tensor_model_parallel_size", 1), ("pipeline_model_parallel_size", 1),
                            ("context_parallel_size", 1), ("fp16", False), ("bf16", False)):
        if getattr(config, field, expected) != expected:
            return "unsupported model setting: " + field
    for field in ("num_moe_experts", "num_experts", "mtp_num_layers", "virtual_pipeline_model_parallel_size"):
        if getattr(config, field, None) not in (None, 0):
            return "unsupported model feature: " + field
    for field in ("multi_latent_attention", "gated_linear_unit"):
        if getattr(config, field, False):
            return "unsupported model feature: " + field
    if getattr(model, "position_embedding_type", "learned_absolute") != "learned_absolute":
        return "unverified position embedding type"
    if not getattr(model, "pre_process", True) or not getattr(model, "post_process", True):
        return "full single-stage GPTModel required"
    if getattr(model, "fp16_lm_cross_entropy", False):
        return "FP16 loss mode is not verified"
    layers = getattr(getattr(model, "decoder", None), "layers", None)
    if layers is None or len(layers) != config.num_layers:
        return "decoder layer count mismatch"
    if not callable(getattr(model, "compute_language_model_loss", None)):
        return "missing native language-model loss"
    for layer in layers:
        norm = getattr(layer, "pre_mlp_layernorm", None)
        if not isinstance(norm, torch.nn.RMSNorm):
            return "local torch.nn.RMSNorm required; TE and other norm implementations are not verified"
        if not callable(getattr(layer, "self_attn_bda", None)):
            return "missing native attention BDA factory"
        if tuple(norm.weight.shape) != (config.hidden_size,) or norm.eps != 1e-5:
            return "RMSNorm parameters differ from the verified configuration"
    return None


def _tensor_metadata(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "stride": list(value.stride())}


class InstallationHandle:
    """Own the instance wrappers and emit tensor-free audit metadata."""

    def __init__(self, enabled, model=None, candidate=None, provenance=None):
        self.enabled = bool(enabled)
        self.active = False
        self._model = model if enabled else None
        self._candidate = candidate
        self._provenance = provenance or {}
        self._mutations = []
        self._pending = []
        self._devices = {}
        self._world_size = None
        self._model_type = None if model is None else type(model).__module__ + "." + type(model).__name__
        config = getattr(model, "config", None)
        self._config = {key: getattr(config, key, None) for key in _CONFIG_FIELDS} if config else {}
        self._counts = {
            "residual": {"candidate_api_calls": 0, "cached_norm_calls": 0, "native_bda_calls": 0,
                         "native_norm_calls": 0, "autocast_compatibility_calls": 0,
                         "fallback_reasons": {}, "observations": []},
            "cross_entropy": {"candidate_api_calls": 0, "native_calls": 0,
                              "fp32_triton_eligible_calls": 0, "autocast_torch_compatibility_calls": 0,
                              "fallback_reasons": {}, "observations": []},
        }

    def _replace(self, obj, name, replacement):
        existing = vars(obj)
        self._mutations.append((obj, name, name in existing, existing.get(name), replacement))
        setattr(obj, name, replacement)

    def _fallback(self, operation, reason):
        counts = self._counts[operation]
        key = "native_bda_calls" if operation == "residual" else "native_calls"
        counts[key] += 1
        counts["fallback_reasons"][reason] = counts["fallback_reasons"].get(reason, 0) + 1

    def _device_reason(self, *tensors):
        if not all(getattr(tensor, "is_cuda", False) for tensor in tensors):
            return "non_cuda_tensors"
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            return "mixed_devices"
        if self._world_size is None:
            distributed = self._candidate.torch.distributed
            if not distributed.is_initialized():
                return "distributed_not_initialized"
            self._world_size = distributed.get_world_size()
        if self._world_size != 1:
            return "unverified_world_size"
        index = device.index
        if index != 0:
            return "unverified_device_index"
        if index not in self._devices:
            self._devices[index] = str(self._candidate.torch.cuda.get_device_name(device))
        if self._devices[index] not in ("Iluvatar BI-V150", "BI-V150"):
            return "unverified_gpu_architecture"
        return None

    def _autocast(self):
        return bool(self._candidate._cuda_autocast_enabled())

    def snapshot(self):
        return deepcopy({
            "enabled": self.enabled, "active": self.active,
            "provenance_verified": bool(self._provenance), **self._provenance,
            "candidate_expected_sha256": CANDIDATE_SHA256,
            "model_type": self._model_type, "config_at_installation": self._config,
            "verified_devices": {str(key): value for key, value in self._devices.items()},
            "runtime_world_size": self._world_size,
            "counts": self._counts,
            "scope": "Fixed small MCore GPT configurations; world/TP/PP/CP=1; tested runtime shapes only.",
            "count_scope": "API dispatch and native fallback counters. Eligible calls are not GPU launch evidence.",
            "autocast_scope": "Residual native fallback and native-equivalent Torch CE; no Triton CE fusion under autocast.",
        })

    def uninstall(self):
        if not self.active:
            return False
        for obj, name, _, _, replacement in self._mutations:
            if getattr(obj, name, None) is not replacement:
                raise AdapterError("Another hook changed an adapter-owned attribute: " + name)
        for obj, name, had_attribute, previous, _ in reversed(self._mutations):
            if had_attribute:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        for pending in self._pending:
            pending.clear()
        self._mutations.clear()
        self._pending.clear()
        self._model = None
        self.active = False
        return True


def _install_residual_layer(handle, layer):
    norm = layer.pre_mlp_layernorm
    original_factory, original_norm = layer.self_attn_bda, norm.forward
    pending = {}
    handle._pending.append(pending)

    def factory(training, fused):
        original = original_factory(training, fused)

        def bda(values, residual, dropout):
            pending.clear()
            x, bias = values
            reason = None
            if not training:
                reason = "not_training"
            elif fused:
                reason = "native_bias_dropout_fusion_requested"
            elif dropout != 0 or bias is not None:
                reason = "dropout_or_bias"
            elif x.dim() not in (2, 3) or tuple(x.shape) != tuple(residual.shape):
                reason = "shape"
            elif x.shape[-1] < 1 or (x.numel() // x.shape[-1], x.shape[-1]) not in _RESIDUAL_SHAPES:
                reason = "unverified_shape"
            elif norm.eps != 1e-5:
                reason = "epsilon"
            elif (residual.dtype != handle._candidate.torch.float32 or
                  norm.weight.dtype != handle._candidate.torch.float32 or
                  x.dtype not in (handle._candidate.torch.float32, handle._candidate.torch.float16,
                                  handle._candidate.torch.bfloat16)):
                reason = "dtype"
            elif not all(tensor.is_contiguous() for tensor in (x, residual, norm.weight)):
                reason = "layout"
            else:
                reason = handle._device_reason(x, residual, norm.weight)
            if reason is not None:
                handle._fallback("residual", reason)
                return original(values, residual, dropout)
            # Do not catch candidate failures: CE and native CE can mutate input
            # storage, so retrying a partially executed call is not a safe fallback.
            y, summed = handle._candidate.fused(x, residual, norm.weight, norm.eps)
            pending["result"] = (summed, y)
            counts = handle._counts["residual"]
            counts["candidate_api_calls"] += 1
            if handle._autocast():
                counts["autocast_compatibility_calls"] += 1
            if len(counts["observations"]) < 4:
                counts["observations"].append({"x": _tensor_metadata(x), "residual": _tensor_metadata(residual),
                                               "normalized": _tensor_metadata(y), "epsilon": norm.eps})
            return summed

        return bda

    def norm_forward(self, hidden):
        result = pending.pop("result", None)
        if result is not None and hidden is result[0]:
            handle._counts["residual"]["cached_norm_calls"] += 1
            return result[1]
        handle._counts["residual"]["native_norm_calls"] += 1
        return original_norm(hidden)

    handle._replace(layer, "self_attn_bda", factory)
    handle._replace(norm, "forward", types.MethodType(norm_forward, norm))


def _install_cross_entropy(handle, model):
    original = model.compute_language_model_loss

    def compute_loss(self, labels, logits):
        reason = None
        if not self.training:
            reason = "not_training"
        elif self.config.cross_entropy_loss_fusion:
            reason = "native_loss_fusion_requested"
        elif self.tp_group is None or self.tp_group.size() != 1:
            reason = "tensor_parallel_group"
        elif logits.dim() != 3 or labels.dim() != 2:
            reason = "rank"
        elif tuple(logits.shape) not in _CE_SHAPES:
            reason = "unverified_shape_or_vocabulary"
        elif tuple(labels.shape) != (logits.shape[1], logits.shape[0]):
            reason = "label_shape"
        elif labels.dtype != handle._candidate.torch.int64:
            reason = "target_dtype"
        elif logits.dtype not in (handle._candidate.torch.float32, handle._candidate.torch.float16,
                                   handle._candidate.torch.bfloat16):
            reason = "logits_dtype"
        elif not logits.is_contiguous():
            reason = "logits_layout"
        else:
            reason = handle._device_reason(logits, labels)
        if reason is not None:
            handle._fallback("cross_entropy", reason)
            return original(labels, logits)
        target = labels.transpose(0, 1).contiguous()
        loss = handle._candidate.cross_entropy(logits, target)
        counts = handle._counts["cross_entropy"]
        counts["candidate_api_calls"] += 1
        if handle._autocast():
            counts["autocast_torch_compatibility_calls"] += 1
        else:
            counts["fp32_triton_eligible_calls"] += 1
        if len(counts["observations"]) < 4:
            counts["observations"].append({"logits": _tensor_metadata(logits), "target": _tensor_metadata(target),
                                           "loss": _tensor_metadata(loss)})
        return loss.transpose(0, 1).contiguous()

    handle._replace(model, "compute_language_model_loss", types.MethodType(compute_loss, model))


def install_kernels(model, *, enabled=False, megatron_root=None, candidate_path=None):
    """Install once on an unwrapped model; disabled means a strict no-op.

    The launcher should retain the returned handle and require nonzero candidate
    coverage before reporting an optimized run. Unsupported runtime calls remain
    native and are counted. This function performs no GPU tensor allocation.
    """
    existing = getattr(model, _MARKER, None)
    if not enabled:
        if existing is not None:
            raise AdapterError("An installed model cannot be labeled baseline; uninstall or construct a fresh model")
        return InstallationHandle(False, model=model)
    if megatron_root is None:
        raise AdapterError("Opt-in installation requires the fixed Megatron checkout path")
    provenance = _verify_megatron(megatron_root)
    path = DEFAULT_CANDIDATE if candidate_path is None else Path(candidate_path)
    candidate = _load_candidate(path)
    provenance.update(candidate_sha256=CANDIDATE_SHA256, candidate_path=str(path.resolve()),
                      torch=str(candidate.torch.__version__), triton=str(candidate.triton.__version__))
    reason = _model_reason(model, candidate.torch)
    if reason is not None:
        raise AdapterError(reason)
    if existing is not None:
        if isinstance(existing, InstallationHandle) and existing.active:
            return existing
        raise AdapterError("An unknown adapter marker already exists on this model")
    handle = InstallationHandle(True, model, candidate, provenance)
    try:
        for layer in model.decoder.layers:
            _install_residual_layer(handle, layer)
        _install_cross_entropy(handle, model)
        handle._replace(model, _MARKER, handle)
        handle.active = True
    except Exception:
        # All bindings are installed before model execution, so rollback here
        # cannot retry or mutate any candidate tensor operation.
        for obj, name, had_attribute, previous, _ in reversed(handle._mutations):
            if had_attribute:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        raise
    return handle
