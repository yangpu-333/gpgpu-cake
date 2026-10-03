"""Metadata-only probes for an unwrapped native Megatron GPT instance.

Install from its pre_wrap_hook. Native call arguments and return objects pass
through unchanged; no tensor values are copied, retained, reduced, synchronized,
or moved. These probes are for separate audit/profile runs, not throughput runs.
The record_function annotations describe host call scopes. GPU timing must come
from an independently exported real profiler trace.
"""
from contextlib import nullcontext
from copy import deepcopy
import types


_MARKER = "_kernelwiki_native_target_profile"


class ProfileHookError(RuntimeError):
    pass


def _source(function):
    return {"module": getattr(function, "__module__", type(function).__module__),
            "name": getattr(function, "__qualname__", type(function).__qualname__)}


def _argument(args, kwargs, index, *names):
    if len(args) > index:
        return args[index]
    for name in names:
        if name in kwargs:
            return kwargs[name]
    return None


def _primitive(value):
    if type(value).__module__ == "torch" and type(value).__name__ == "dtype":
        return str(value)
    return value if value is None or type(value) in (bool, int, float, str) else {
        "type": type(value).__module__ + "." + type(value).__qualname__}


class ProfileHandle:
    """Own instance bindings and JSON-safe observations without tensor storage."""

    def __init__(self, model, torch_module, max_observations, record_ranges):
        self._model = model
        self._torch = torch_module
        self._max_observations = max_observations
        self._record_ranges = bool(record_ranges)
        self._mutations = []
        self._hook_handles = []
        self.active = False
        self._forward_calls = 0
        self._completed_forwards = 0
        self._current_forward = None
        self._metadata_errors = []
        self._layers = []
        self._ce = self._operation()
        cfg = getattr(model, "config", None)
        self._config = {name: _primitive(getattr(cfg, name, None)) for name in (
            "num_layers", "hidden_size", "num_attention_heads", "ffn_hidden_size", "normalization",
            "layernorm_epsilon", "bf16", "fp16", "params_dtype", "fp32_residual_connection",
            "hidden_dropout", "attention_dropout", "cross_entropy_loss_fusion",
            "tensor_model_parallel_size", "pipeline_model_parallel_size", "context_parallel_size")}
        self._model_class = type(model).__module__ + "." + type(model).__qualname__

    @staticmethod
    def _operation():
        return {"forward_calls": 0, "completed_calls": 0, "failed_calls": 0, "observations": []}

    def _metadata(self, value):
        if value is None:
            return None
        if not self._torch.is_tensor(value):
            return {"type": type(value).__module__ + "." + type(value).__qualname__, "is_tensor": False}
        try:
            return {"dtype": str(value.dtype), "shape": [int(size) for size in value.shape],
                    "stride": [int(stride) for stride in value.stride()], "device": str(value.device),
                    "layout": str(value.layout), "requires_grad": bool(value.requires_grad)}
        except Exception as error:
            # An observation must never turn a successful native call into a
            # failure, e.g. for a tensor layout whose stride is unavailable.
            error_type = type(error).__name__
            if len(self._metadata_errors) < self._max_observations:
                self._metadata_errors.append(error_type)
            return {"is_tensor": True, "metadata_error_type": error_type}

    def _range(self, label):
        if not self._record_ranges:
            return nullcontext()
        return self._torch.profiler.record_function(label)

    def _replace(self, obj, name, replacement):
        namespace = vars(obj)
        self._mutations.append((obj, name, name in namespace, namespace.get(name), replacement))
        setattr(obj, name, replacement)

    def _call(self, operation, label, original, args, kwargs, inputs, result_name="result"):
        if not self.active:
            return original(*args, **kwargs)
        operation["forward_calls"] += 1
        observation = None
        if len(operation["observations"]) < self._max_observations:
            observation = {"model_forward_ordinal": self._current_forward,
                           "operation_call_ordinal": operation["forward_calls"], **inputs()}
        try:
            with self._range(label):
                result = original(*args, **kwargs)
        except BaseException as error:
            operation["failed_calls"] += 1
            if observation is not None:
                observation["native_error_type"] = type(error).__name__
                operation["observations"].append(observation)
            raise
        operation["completed_calls"] += 1
        if observation is not None:
            observation[result_name] = self._metadata(result)
            operation["observations"].append(observation)
        return result

    def _install_layer(self, layer, index):
        original_factory = layer.self_attn_bda
        norm = layer.pre_mlp_layernorm
        original_norm = norm.forward
        state = {"layer_index": index, "native_layer_number": _primitive(getattr(layer, "layer_number", None)),
                 "bda_factory_calls": 0, "bda_factory_failures": 0,
                 "factory_observations": [], "residual_bda": self._operation(), "pre_mlp_rmsnorm": self._operation(),
                 "bound_call_sources": {"bda_factory": _source(original_factory), "norm_forward": _source(original_norm)},
                 "range_labels": {"residual_bda": "kernelwiki/profile/layer_%d/residual_bda" % index,
                                  "pre_mlp_rmsnorm": "kernelwiki/profile/layer_%d/pre_mlp_rmsnorm" % index}}
        self._layers.append(state)

        def factory(*args, **kwargs):
            if not self.active:
                return original_factory(*args, **kwargs)
            state["bda_factory_calls"] += 1
            training = _argument(args, kwargs, 0, "training")
            fused = _argument(args, kwargs, 1, "fused")
            try:
                native_bda = original_factory(*args, **kwargs)
            except BaseException:
                state["bda_factory_failures"] += 1
                raise
            if len(state["factory_observations"]) < self._max_observations:
                state["factory_observations"].append({"model_forward_ordinal": self._current_forward,
                    "factory_call_ordinal": state["bda_factory_calls"], "training": _primitive(training),
                    "fused": _primitive(fused), "returned_native_callable": _source(native_bda)})

            def bda(*call_args, **call_kwargs):
                def inputs():
                    values = _argument(call_args, call_kwargs, 0, "x_with_bias", "values")
                    valid_pair = isinstance(values, (tuple, list)) and len(values) == 2
                    x, bias = values if valid_pair else (None, None)
                    return {"x": self._metadata(x), "bias": self._metadata(bias),
                            "residual": self._metadata(_argument(call_args, call_kwargs, 1, "residual")),
                            "dropout_probability": _primitive(_argument(call_args, call_kwargs, 2, "prob", "dropout")),
                            "factory_training": _primitive(training), "factory_fused": _primitive(fused)}
                return self._call(state["residual_bda"], state["range_labels"]["residual_bda"],
                                  native_bda, call_args, call_kwargs, inputs)
            return bda

        def norm_forward(module, *args, **kwargs):
            def inputs():
                return {"input": self._metadata(_argument(args, kwargs, 0, "input", "x", "hidden")),
                        "weight": self._metadata(getattr(module, "weight", None)),
                        "epsilon": _primitive(getattr(module, "eps", None))}
            return self._call(state["pre_mlp_rmsnorm"], state["range_labels"]["pre_mlp_rmsnorm"],
                              original_norm, args, kwargs, inputs, result_name="output")

        self._replace(layer, "self_attn_bda", factory)
        self._replace(norm, "forward", types.MethodType(norm_forward, norm))

    def _install_ce(self):
        original = self._model.compute_language_model_loss
        self._ce["bound_call_source"] = _source(original)
        self._ce["range_label"] = "kernelwiki/profile/native_cross_entropy"

        def compute_loss(module, *args, **kwargs):
            def inputs():
                return {"labels": self._metadata(_argument(args, kwargs, 0, "labels")),
                        "logits": self._metadata(_argument(args, kwargs, 1, "logits"))}
            return self._call(self._ce, self._ce["range_label"], original, args, kwargs, inputs, result_name="loss")
        self._replace(self._model, "compute_language_model_loss", types.MethodType(compute_loss, self._model))

    def snapshot(self):
        return deepcopy({"schema": "megatron-native-target-profile-v1", "active": self.active,
                         "model_class": self._model_class, "config_at_installation": self._config,
                         "model_forward_calls": self._forward_calls,
                         "model_completed_forwards": self._completed_forwards,
                         "max_observations_per_operation": self._max_observations,
                         "record_function_ranges_enabled": self._record_ranges,
                         "layers": self._layers, "native_cross_entropy": self._ce,
                         "metadata_error_types": self._metadata_errors,
                         "observation_scope": "Only dtype/shape/stride/device/layout/requires_grad and scalar call metadata. "
                                              "No tensor values or tensor objects are retained. Forward ordinals are not training iteration IDs.",
                         "timing_scope": "CPU record_function call-scope annotations only; no timing is measured by this hook. "
                                         "GPU kernel attribution requires the separately exported real profiler trace. "
                                         "Profiler/audit overhead must not be included in throughput comparisons."})

    def uninstall(self):
        if not self.active:
            return False
        for obj, name, _, _, replacement in self._mutations:
            if getattr(obj, name, None) is not replacement:
                raise ProfileHookError("Another hook changed this probe-owned attribute: " + name)
        for hook in self._hook_handles:
            hook.remove()
        for obj, name, present, previous, _ in reversed(self._mutations):
            if present:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        self._hook_handles.clear()
        self._mutations.clear()
        self._model = None
        self._current_forward = None
        self.active = False
        return True


def install_profile_hooks(model, *, max_observations=8, record_ranges=True, torch_module=None):
    """Install on the original GPT model before precision/distributed wrappers."""
    if type(max_observations) is not int or max_observations < 0:
        raise ValueError("max_observations must be a nonnegative integer")
    existing = getattr(model, _MARKER, None)
    if existing is not None:
        if isinstance(existing, ProfileHandle) and existing.active:
            return existing
        raise ProfileHookError("A different target probe marker is already installed")
    layers = list(model.decoder.layers)
    if not layers or not callable(getattr(model, "compute_language_model_loss", None)):
        raise ProfileHookError("An unwrapped GPT model with decoder layers and native loss is required")
    norms = []
    for layer in layers:
        norm = getattr(layer, "pre_mlp_layernorm", None)
        if not callable(getattr(layer, "self_attn_bda", None)) or not callable(getattr(norm, "forward", None)):
            raise ProfileHookError("Each layer must expose native residual BDA and pre-MLP norm")
        norms.append(norm)
    if len({id(norm) for norm in norms}) != len(norms):
        raise ProfileHookError("Shared pre-MLP norm instances need a separate unambiguous probe design")
    if torch_module is None:
        import torch as torch_module
    handle = ProfileHandle(model, torch_module, max_observations, record_ranges)
    try:
        for index, layer in enumerate(layers):
            handle._install_layer(layer, index)
        handle._install_ce()
        def before_forward(module, inputs):
            if handle.active:
                handle._forward_calls += 1
                handle._current_forward = handle._forward_calls
        def after_forward(module, inputs, output):
            if handle.active:
                handle._completed_forwards += 1
                handle._current_forward = None
        handle._hook_handles.append(model.register_forward_pre_hook(before_forward))
        handle._hook_handles.append(model.register_forward_hook(after_forward))
        handle._replace(model, _MARKER, handle)
        handle.active = True
    except BaseException:
        for hook in handle._hook_handles:
            hook.remove()
        for obj, name, present, previous, _ in reversed(handle._mutations):
            if present:
                setattr(obj, name, previous)
            else:
                delattr(obj, name)
        raise
    return handle
