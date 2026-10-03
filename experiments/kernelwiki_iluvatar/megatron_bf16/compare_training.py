"""Compare immutable three-step BF16 formal Megatron audit cases on CPU.

This reads only tensor snapshots emitted by launch_training.py and the pinned
native save_grads routine. It never imports Megatron, candidate kernels, or
training code. BF16 model parameters and FP32 optimizer master parameters are
both checked; BF16 rounding cannot conceal an optimizer update mismatch.
Full training checkpoints are inventoried but not unpickled.
"""
import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ATOL = 3e-4
RTOL = 1e-3
STEPS = 3
COMMIT = "5be9626709af2722333bf54797c954c09edeada3"
ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Comparison:
    def __init__(self, native, optimized, torch_module, contract_path=None, candidate_path=None):
        self.native, self.optimized = Path(native), Path(optimized)
        self.torch = torch_module
        self.checks, self.tensors, self.raw_values, self.files = [], [], [], []
        self.tensor_groups = {}
        self.coverage = {}
        self.derived_losses = []
        self.model = {}
        self.parameter_names = {}
        self.initial_dtype_counts = {}
        self.contract_path = Path(contract_path) if contract_path is not None else None
        self.candidate_path = Path(candidate_path) if candidate_path is not None else None

    def require(self, condition, name, **details):
        passed = bool(condition)
        self.checks.append({"name": name, "passed": passed, **details})
        return passed

    def inventory(self, path, arm):
        path = Path(path)
        self.files.append({"arm": arm, "path": str(path.resolve()),
                           "bytes": path.stat().st_size, "sha256": sha256(path)})

    def read_json(self, arm, name):
        directory = self.native if arm == "native" else self.optimized
        path = directory / name
        if not self.require(path.is_file(), "%s/%s exists" % (arm, name)):
            return None
        self.inventory(path, arm)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            self.require(False, "%s/%s readable JSON" % (arm, name), error=str(error))
            return None
        if not self.require(isinstance(value, dict), "%s/%s is an object" % (arm, name)):
            return None
        return value

    def read_snapshot(self, arm, relative):
        directory = self.native if arm == "native" else self.optimized
        path = directory / "snapshots" / relative
        if not self.require(path.is_file(), "%s/%s exists" % (arm, relative)):
            return None
        self.inventory(path, arm)
        try:
            # These experiment-owned files contain tensors, built-in mappings,
            # and None only. Do not silently enable arbitrary pickle loading.
            return self.torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            self.require(False, "%s/%s safely readable snapshot" % (arm, relative),
                         error_type=type(error).__name__, error=str(error))
            return None

    def tensor_digest(self, value):
        data = value.detach().cpu().contiguous().reshape(-1)
        return hashlib.sha256(data.view(self.torch.uint8).numpy().tobytes()).hexdigest()

    def compare_tensor(self, native, optimized, name, group, exact=False):
        torch = self.torch
        record = {"path": name, "group": group, "exact_required": exact,
                  "native_shape": list(native.shape), "optimized_shape": list(optimized.shape),
                  "native_dtype": str(native.dtype), "optimized_dtype": str(optimized.dtype),
                  "native_numel": native.numel(), "optimized_numel": optimized.numel()}
        self.tensor_groups[group] = self.tensor_groups.get(group, 0) + 1
        supported = (native.layout == torch.strided and optimized.layout == torch.strided
                     and not native.is_complex() and not optimized.is_complex())
        compatible = supported and native.shape == optimized.shape and native.dtype == optimized.dtype
        if supported:
            record.update(native_sha256=self.tensor_digest(native),
                          optimized_sha256=self.tensor_digest(optimized))
        record["shape_dtype_layout_match"] = compatible
        if compatible:
            left, right = native.detach().cpu(), optimized.detach().cpu()
            finite = bool(torch.isfinite(left).all().item() and torch.isfinite(right).all().item())
            same = bool(torch.equal(left, right))
            record.update(all_elements_finite=finite, exact_equal=same)
            if finite:
                a, b = left.to(torch.float64), right.to(torch.float64)
                error = (b - a).abs()
                allowed = ATOL + RTOL * a.abs()
                outside = error > allowed
                record.update(max_absolute_error=float(error.max().item()) if error.numel() else 0.0,
                              max_relative_error_to_clamped_native=float(
                                  (error / a.abs().clamp_min(ATOL)).max().item()) if error.numel() else 0.0,
                              max_tolerance_fraction=float((error / allowed).max().item()) if error.numel() else 0.0,
                              elements_outside_tolerance=int(outside.sum().item()),
                              elements_not_exactly_equal=int((left != right).sum().item()),
                              atol=0.0 if exact else ATOL, rtol=0.0 if exact else RTOL)
                passed = same if exact else not bool(outside.any().item())
            else:
                passed = False
        else:
            record["all_elements_finite"] = False
            passed = False
        record["passed"] = passed
        self.tensors.append(record)
        self.require(passed, name)

    def compare_tree(self, native, optimized, name, group, exact=False):
        torch = self.torch
        left_tensor, right_tensor = torch.is_tensor(native), torch.is_tensor(optimized)
        if left_tensor or right_tensor:
            if not self.require(left_tensor and right_tensor, name + " tensor types match"):
                return
            self.compare_tensor(native, optimized, name, group, exact)
        elif isinstance(native, Mapping) or isinstance(optimized, Mapping):
            if not self.require(isinstance(native, Mapping) and isinstance(optimized, Mapping),
                                name + " mapping types match"):
                return
            left_keys, right_keys = set(native), set(optimized)
            self.require(left_keys == right_keys, name + " keys match",
                         native_only=sorted(map(str, left_keys - right_keys)),
                         optimized_only=sorted(map(str, right_keys - left_keys)))
            for key in sorted(left_keys & right_keys, key=str):
                self.compare_tree(native[key], optimized[key], name + "/" + str(key), group, exact)
        elif isinstance(native, (list, tuple)) or isinstance(optimized, (list, tuple)):
            if not self.require(type(native) is type(optimized) and len(native) == len(optimized),
                                name + " sequence types and lengths match"):
                return
            for index, (left, right) in enumerate(zip(native, optimized)):
                self.compare_tree(left, right, name + "/" + str(index), group, exact)
        else:
            permitted = (type(None), bool, str, int, float)
            supported = type(native) in permitted and type(optimized) in permitted
            finite = (not isinstance(native, float) or math.isfinite(native)) and (
                not isinstance(optimized, float) or math.isfinite(optimized))
            same = supported and finite and type(native) is type(optimized) and native == optimized
            self.raw_values.append({"path": name, "native_type": type(native).__name__,
                                    "optimized_type": type(optimized).__name__, "passed": same})
            self.require(same, name + " raw values recursively equivalent")

    @staticmethod
    def normalized_argv(argv):
        # Each arm saves in its own immutable directory; all other training
        # arguments (including seed and audit flags) must be identical.
        result, index = [], 0
        while index < len(argv):
            if argv[index] == "--save":
                if index + 1 >= len(argv):
                    raise ValueError("missing --save value")
                result += ["--save", "<case-snapshots>"]
                index += 2
            else:
                result.append(argv[index])
                index += 1
        return result

    def metadata(self):
        if not self.require(self.contract_path is not None and self.candidate_path is not None,
                            "explicit sealed contract and candidate paths supplied"):
            return None
        contract_path = self.contract_path
        manifest_path = ROOT.parent / "megatron_training_entry" / "source_manifest.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.inventory(contract_path, "fixed-verifier")
        self.inventory(manifest_path, "fixed-verifier")
        self.inventory(self.candidate_path, "fixed-verifier")
        correctness = contract.get("correctness", {})
        self.require(correctness.get("atol") == ATOL and correctness.get("rtol") == RTOL
                     and correctness.get("steps") == STEPS,
                     "sealed contract preserves exact FP32 tolerances and three steps")
        self.require(contract.get("megatron_commit") == COMMIT,
                     "sealed contract pins native commit")
        expected_candidate = contract.get("candidate_sha256")
        self.require(isinstance(expected_candidate, str) and len(expected_candidate) == 64
                     and sha256(self.candidate_path) == expected_candidate,
                     "candidate bytes match sealed BF16 contract")
        self.model = contract.get("model", {})
        required = ("num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads",
                    "sequence_length", "micro_batch_size", "global_batch_size", "vocab_size", "parameters")
        self.require(isinstance(self.model, dict) and
                     all(isinstance(self.model.get(name), int) and self.model[name] > 0 for name in required),
                     "sealed model supplies all positive configuration and parameter sizes")
        self.require(self.model.get("bf16") is True and self.model.get("normalization") == "RMSNorm"
                     and self.model.get("epsilon") == 1e-5
                     and self.model.get("fp32_residual_connection") is False,
                     "sealed model is true BF16 with native BF16 residual and RMSNorm")
        self.require(manifest.get("commit") == COMMIT and isinstance(manifest.get("files"), dict)
                     and len(manifest.get("files", {})) == 600, "fixed complete 600-file native manifest present")
        records = {arm: {name: self.read_json(arm, name) for name in ("command.json", "entry.json")}
                   for arm in ("native", "optimized")}
        commands = [records[arm]["command.json"] for arm in ("native", "optimized")]
        entries = [records[arm]["entry.json"] for arm in ("native", "optimized")]
        if any(value is None for value in commands + entries):
            return None
        expected_integration = {name: sha256(ROOT / name) for name in
                                ("launch_training.py", "kernel_adapter.py", "run_case.py")}
        expected_integration.update({"formal/" + name: sha256(manifest_path.parent / name)
                                     for name in ("runtime_compat.py", "source_manifest.json")})
        expected_contract = sha256(contract_path)
        for arm, command, entry in zip(("native", "optimized"), commands, entries):
            self.require(command.get("mode") == arm, arm + " run mode")
            self.require(command.get("audit") is True and command.get("profile") is False
                         and command.get("steps") == STEPS, arm + " three-step unprofiled tensor audit")
            self.require(command.get("exit_code") == 0 and entry.get("status") == "completed",
                         arm + " formal native launch completed")
            self.require(entry.get("megatron_commit") == COMMIT, arm + " pinned Megatron commit")
            self.require(command.get("contract_sha256") == expected_contract
                         and entry.get("contract_sha256") == expected_contract,
                         arm + " command and launch match sealed verifier contract")
            self.require(entry.get("native_sources_sha256") == manifest.get("files"),
                         arm + " all native sources match exact fixed manifest")
            self.require(entry.get("integration_sha256") == expected_integration,
                         arm + " all integration sources match verifier files")
            self.require(entry.get("launcher_sha256") == expected_integration["launch_training.py"],
                         arm + " final launcher remains frozen")
            models = entry.get("models")
            if not self.require(isinstance(models, list) and len(models) == 1,
                                arm + " exactly one native model chunk"):
                continue
            model = models[0]
            if not self.require(isinstance(model, dict), arm + " model metadata object"):
                continue
            for name, expected in {"parameters": self.model.get("parameters"),
                                   "layers": self.model.get("num_layers"),
                                   "hidden_size": self.model.get("hidden_size"),
                                   "vocab_size": self.model.get("vocab_size"),
                                   "bf16": True, "fp16": False,
                                   "class": "megatron.core.models.gpt.gpt_model.GPTModel"}.items():
                self.require(model.get(name) == expected, arm + " fixed builder record " + name)
            names = model.get("parameter_names")
            valid = isinstance(names, list) and bool(names) and all(isinstance(name, str) for name in names)
            valid = valid and len(names) == len(set(names))
            if self.require(valid, arm + " complete unique native parameter names recorded"):
                self.parameter_names[arm] = set(names)
        self.require(commands[0].get("seed") == commands[1].get("seed")
                     and isinstance(commands[0].get("seed"), int), "same explicit seed")
        self.require(entries[0].get("native_sources_sha256") == entries[1].get("native_sources_sha256"),
                     "both arms have identical native sources")
        self.require(entries[0].get("integration_sha256") == entries[1].get("integration_sha256"),
                     "both arms have identical integration sources")
        self.compare_tree(entries[0].get("models"), entries[1].get("models"),
                          "model construction metadata", "metadata", exact=True)
        argv = [entry.get("training_argv") for entry in entries]
        self.require(all(isinstance(value, list) and all(isinstance(item, str) for item in value) for value in argv),
                     "both arms have complete formal training argv")
        if all(isinstance(value, list) for value in argv):
            normalized = [self.normalized_argv(value) for value in argv]
            self.require(normalized[0] == normalized[1], "identical training argv except output directory")
            suffix = ["--train-iters", str(STEPS), "--seed", str(commands[0].get("seed")),
                      "--log-interval", "1", "--save", "<case-snapshots>",
                      "--save-interval", str(STEPS), "--save-wgrads-interval", "1",
                      "--save-params-interval", "1"]
            self.require(isinstance(contract.get("training_argv"), list) and
                         all(value == contract["training_argv"] + suffix for value in normalized),
                         "both arms exactly execute sealed BF16 audit arguments")
        self.require(entries[0].get("iluvatar_kernels") is False and entries[0].get("kernel_adapters") == [],
                     "native has no candidate installation")
        self.require(entries[1].get("iluvatar_kernels") is True, "optimized explicitly opted in")
        self.require(all(entry.get("profile_hooks") == [] for entry in entries),
                     "both correctness arms have no profiler hooks")
        adapters = entries[1].get("kernel_adapters")
        if not self.require(isinstance(adapters, list) and len(adapters) == 1,
                            "optimized has exactly one BF16 CE adapter"):
            return records
        handle = adapters[0]
        if not self.require(isinstance(handle, dict), "adapter metadata is an object"):
            return records
        self.require(all(handle.get(key) is True for key in ("enabled", "active", "provenance_verified")),
                     "optimized adapter active and provenance verified")
        self.require(handle.get("candidate_sha256") == expected_candidate and
                     handle.get("candidate_expected_sha256") == expected_candidate,
                     "adapter selects exact sealed candidate")
        self.require(handle.get("runtime_world_size") == 1, "adapter observed world size one")
        devices = handle.get("verified_devices", {})
        self.require(isinstance(devices, dict) and set(devices) == {"0"} and
                     devices.get("0") in ("BI-V150", "Iluvatar BI-V150"), "adapter observed BI-V150 zero")
        expected_counts = {"ce_candidate_calls": STEPS, "ce_native_calls": 0,
                           "residual_native_calls": self.model.get("num_layers", 0) * STEPS,
                           "fallback_reasons": {}}
        counts = handle.get("counts", {})
        for name, expected in expected_counts.items():
            self.require(isinstance(counts, dict) and counts.get(name) == expected,
                         "BF16 adapter coverage " + name, expected=expected,
                         observed=counts.get(name) if isinstance(counts, dict) else None)
        cfg = handle.get("config_at_installation", {})
        expected_config = {name: self.model.get(name) for name in
                           ("num_layers", "hidden_size", "ffn_hidden_size", "num_attention_heads")}
        expected_config.update({"bf16": True, "fp16": False, "normalization": "RMSNorm",
                                "layernorm_epsilon": 1e-5, "hidden_dropout": 0.0,
                                "attention_dropout": 0.0, "add_bias_linear": False,
                                "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                                "context_parallel_size": 1})
        for name, expected in expected_config.items():
            self.require(isinstance(cfg, dict) and cfg.get(name) == expected,
                         "BF16 adapter fixed configuration " + name)
        observations = handle.get("observations")
        if self.require(isinstance(observations, list) and len(observations) == STEPS,
                        "adapter captures each of three CE dispatch signatures"):
            for index, observed in enumerate(observations, 1):
                expected = {"logits_shape": [self.model.get("sequence_length"),
                                            self.model.get("micro_batch_size"), self.model.get("vocab_size")],
                            "logits_dtype": "torch.bfloat16", "loss_shape": [self.model.get("sequence_length"),
                            self.model.get("micro_batch_size")], "loss_dtype": "torch.float32", "autocast": False}
                self.require(observed == expected, "step %d actual BF16 logits and FP32 CE output signature" % index)
        self.coverage = {"expected": expected_counts, "observed": counts,
                         "scope": "Candidate CE API dispatch at every BF16 forward; residual/RMSNorm stay native. "
                                  "Counters do not independently prove compiled GPU kernel launches."}
        return records

    def snapshot_layout(self):
        expected = {"model0-initial.pt", "actual-start.pt"} | {
            "model0-forward-%03d.pt" % step for step in range(1, STEPS + 1)} | {
            "master-after-%03d.pt" % step for step in range(1, STEPS + 1)}
        for arm, directory in (("native", self.native), ("optimized", self.optimized)):
            snapshots = directory / "snapshots"
            observed = {path.name for path in snapshots.glob("*.pt")}
            self.require(observed == expected, arm + " exact initial/forward/master snapshot inventory",
                         expected=sorted(expected), observed=sorted(observed))
            for label in ("wgrads", "params"):
                observed = {str(path.relative_to(snapshots / label)).replace("\\", "/")
                            for path in (snapshots / label).glob("iter_*/*") if path.is_file()}
                expected_dumps = {"iter_%07d/mp_rank_00.pth" % step for step in range(1, STEPS + 1)}
                self.require(observed == expected_dumps, arm + " exact native " + label + " inventory",
                             expected=sorted(expected_dumps), observed=sorted(observed))

    def derived_loss(self, native, optimized, step):
        torch = self.torch
        values = []
        masks = [snapshot.get("keyword_inputs", {}).get("loss_mask")
                 if isinstance(snapshot, Mapping) else None for snapshot in (native, optimized)]
        if not any(torch.is_tensor(mask) for mask in masks):
            self.require(False, "step %d loss mask captured for derived training loss" % step)
            self.derived_losses.append({"step": step, "available": False,
                                       "scope": "Loss mask was not captured as a model input in either arm; "
                                                "no scalar training loss is inferred. Captured per-token outputs "
                                                "are still compared in full."})
            return
        for arm, snapshot in (("native", native), ("optimized", optimized)):
            output = snapshot.get("output") if isinstance(snapshot, Mapping) else None
            keywords = snapshot.get("keyword_inputs", {}) if isinstance(snapshot, Mapping) else {}
            mask = keywords.get("loss_mask") if isinstance(keywords, Mapping) else None
            valid = torch.is_tensor(output) and torch.is_tensor(mask) and output.shape == mask.shape
            if not self.require(valid, "%s step %d masked loss inputs" % (arm, step)):
                return
            left, weight = output.to(torch.float64), mask.to(torch.float64)
            tokens = weight.sum()
            valid = bool(torch.isfinite(left).all().item() and torch.isfinite(weight).all().item()
                         and torch.isfinite(tokens).item() and tokens.item() > 0)
            if not self.require(valid, "%s step %d finite positive loss-mask token count" % (arm, step)):
                return
            values.append(((left * weight).sum() / tokens, float(tokens.item())))
        self.require(values[0][1] == values[1][1], "step %d identical masked token count" % step)
        self.compare_tensor(values[0][0], values[1][0], "step %d/derived masked mean loss" % step, "derived_loss")
        self.derived_losses.append({"step": step, "native": float(values[0][0].item()),
                                    "optimized": float(values[1][0].item()), "num_tokens": values[0][1],
                                    "scope": "CPU float64 masked mean derived from captured native model outputs. "
                                             "Not a separately intercepted training loss callback."})

    def parameter_mapping(self, values, name, dtype, metadata):
        """Require every parameter, including optimizer masters, by native name."""
        for arm in ("native", "optimized"):
            chunk = values.get(arm)
            if not self.require(isinstance(chunk, Mapping) and bool(chunk),
                                arm + "/" + name + " nonempty parameter mapping"):
                continue
            all_tensors = all(self.torch.is_tensor(value) for value in chunk.values())
            self.require(all_tensors, arm + "/" + name + " contains only parameter tensors")
            self.require(all_tensors and all(value.dtype == dtype for value in chunk.values()),
                         arm + "/" + name + " parameter dtype " + str(dtype))
            expected_names = self.parameter_names.get(arm)
            self.require(isinstance(expected_names, set) and set(chunk) == expected_names,
                         arm + "/" + name + " covers every recorded parameter name",
                         missing=sorted(expected_names - set(chunk)) if isinstance(expected_names, set) else None,
                         extra=sorted(set(chunk) - expected_names) if isinstance(expected_names, set) else None)
            if all_tensors:
                numel = sum(value.numel() for value in chunk.values())
                expected_numel = self.model.get("parameters")
                self.require(isinstance(expected_numel, int) and numel == expected_numel,
                             arm + "/" + name + " covers every recorded parameter element",
                             expected_numel=expected_numel, observed_numel=numel)
            if metadata and all_tensors:
                models = metadata[arm]["entry.json"].get("models", [])
                expected_numel = models[0].get("parameters") if len(models) == 1 else None
                self.require(sum(value.numel() for value in chunk.values()) == expected_numel,
                             arm + "/" + name + " covers builder parameter element count")

    def actual_start(self, initial, metadata):
        values = {arm: self.read_snapshot(arm, "actual-start.pt") for arm in ("native", "optimized")}
        expected = {"model_parameters", "master_parameters"}
        valid = all(isinstance(value, Mapping) and set(value) == expected for value in values.values())
        if not self.require(valid, "actual-start exact model/master mapping keys"):
            return {}
        maps = {label: {arm: values[arm][label] for arm in ("native", "optimized")}
                for label in expected}
        for label, dtype in (("model_parameters", self.torch.bfloat16),
                             ("master_parameters", self.torch.float32)):
            self.parameter_mapping(maps[label], "actual-start/" + label, dtype, metadata)
            self.compare_tree(maps[label]["native"], maps[label]["optimized"],
                              "actual-start/" + label, "actual_start_" + label, exact=True)
        for arm in ("native", "optimized"):
            model, master = maps["model_parameters"][arm], maps["master_parameters"][arm]
            if not isinstance(model, Mapping) or not isinstance(master, Mapping):
                continue
            self.require(set(model) == set(master), arm + "/actual-start model and master names match")
            for name in sorted(set(model) & set(master)):
                if not all(self.torch.is_tensor(value) for value in (model[name], master[name])):
                    continue
                self.compare_tensor(model[name].to(self.torch.float32), master[name],
                                    arm + "/actual-start BF16-to-FP32 master initialization/" + name,
                                    "master_initialization", exact=True)
                if isinstance(initial.get(arm), Mapping) and self.torch.is_tensor(initial[arm].get(name)):
                    self.compare_tensor(initial[arm][name].to(self.torch.bfloat16), model[name],
                                        arm + "/actual-start native BF16 initialization/" + name,
                                        "model_initialization", exact=True)
                else:
                    self.require(False, arm + "/actual-start parameter appears in pre-wrap state_dict/" + name)
        return values

    def native_parameter_dump(self, native, optimized, initial, metadata, step, label):
        expected_keys = {"model_chunk0"}
        valid = all(isinstance(value, Mapping) and set(value) == expected_keys for value in (native, optimized))
        if not self.require(valid, "step %d/%s exactly model_chunk0" % (step, label)):
            return
        chunks = [value["model_chunk0"] for value in (native, optimized)]
        if not self.require(all(isinstance(value, Mapping) and bool(value) for value in chunks),
                            "step %d/%s nonempty parameter mapping" % (step, label)):
            return
        dtype = self.torch.float32 if label == "wgrads" else self.torch.bfloat16
        self.parameter_mapping(dict(zip(("native", "optimized"), chunks)),
                               "step %d/%s" % (step, label), dtype, metadata)
        for arm, chunk in zip(("native", "optimized"), chunks):
            if isinstance(initial[arm], Mapping):
                self.require(set(chunk) <= set(initial[arm]),
                             "%s step %d/%s parameter names occur in initial state_dict" % (arm, step, label))
        self.compare_tree(native, optimized, "step %d/%s" % (step, label), label)

    def master_after(self, step, metadata, parameter_keys, parameter_values):
        values = {arm: self.read_snapshot(arm, "master-after-%03d.pt" % step)
                  for arm in ("native", "optimized")}
        self.parameter_mapping(values, "step %d/FP32 optimizer master" % step,
                               self.torch.float32, metadata)
        if all(isinstance(value, Mapping) for value in values.values()):
            self.compare_tree(values["native"], values["optimized"],
                              "step %d/FP32 optimizer master" % step, "masters")
            for arm, value in values.items():
                expected = parameter_keys.get("params", {}).get(arm)
                self.require(isinstance(expected, set) and set(value) == expected,
                             "%s step %d master and model parameter names match" % (arm, step))
                saved = parameter_values.get("params", {}).get(arm)
                if isinstance(saved, Mapping):
                    saved = saved.get("model_chunk0", {})
                if isinstance(saved, Mapping):
                    for name in sorted(set(value) & set(saved)):
                        if self.torch.is_tensor(value[name]) and self.torch.is_tensor(saved[name]):
                            self.compare_tensor(value[name].to(self.torch.bfloat16), saved[name],
                                                "%s step %d master-to-BF16 model update/%s" % (arm, step, name),
                                                "master_model_consistency", exact=True)

    def run(self):
        metadata = self.metadata()
        self.snapshot_layout()
        initial = {arm: self.read_snapshot(arm, "model0-initial.pt") for arm in ("native", "optimized")}
        self.require(all(isinstance(value, Mapping) and bool(value) for value in initial.values()),
                     "both initial state_dict snapshots are nonempty mappings")
        if all(value is not None for value in initial.values()):
            self.compare_tree(initial["native"], initial["optimized"], "initial state_dict", "initial", exact=True)
            for arm, value in initial.items():
                if isinstance(value, Mapping):
                    counts = {}
                    for item in value.values():
                        if self.torch.is_tensor(item):
                            dtype = str(item.dtype)
                            counts[dtype] = counts.get(dtype, 0) + 1
                    self.initial_dtype_counts[arm] = counts
                    self.require(all(not self.torch.is_tensor(item) or item.dtype in
                                     (self.torch.bfloat16, self.torch.float32)
                                     for item in value.values()),
                                 arm + " pre-wrap initial state preserves native BF16/FP32 construction")
            if metadata:
                for arm in ("native", "optimized"):
                    models = metadata[arm]["entry.json"].get("models", [])
                    if len(models) == 1 and isinstance(initial[arm], Mapping):
                        self.require(list(initial[arm]) == models[0].get("state_dict_keys"),
                                     arm + " initial state_dict keys match builder record")
        self.actual_start(initial, metadata)
        for step in range(1, STEPS + 1):
            forward = {arm: self.read_snapshot(arm, "model0-forward-%03d.pt" % step)
                       for arm in ("native", "optimized")}
            if all(isinstance(value, Mapping) for value in forward.values()):
                expected = {"output", "positional_inputs", "keyword_inputs"}
                self.require(all(set(value) == expected for value in forward.values()),
                             "step %d exact forward snapshot keys" % step)
                for inputs in ("positional_inputs", "keyword_inputs"):
                    self.compare_tree(forward["native"].get(inputs), forward["optimized"].get(inputs),
                                      "step %d/%s" % (step, inputs), "batches", exact=True)
                self.require(all(self.torch.is_tensor(value.get("output")) for value in forward.values()),
                             "step %d model output tensor captured" % step)
                self.require(all(self.torch.is_tensor(value.get("output")) and
                                 value["output"].dtype == self.torch.float32 and
                                 tuple(value["output"].shape) == (self.model.get("micro_batch_size"),
                                                                self.model.get("sequence_length"))
                                 for value in forward.values()),
                             "step %d complete FP32 per-token model loss shape" % step)
                self.compare_tree(forward["native"].get("output"), forward["optimized"].get("output"),
                                  "step %d/output" % step, "output")
                self.derived_loss(forward["native"], forward["optimized"], step)
            else:
                self.require(False, "step %d both forward snapshots are mappings" % step)
            parameter_keys = {}
            parameter_values = {}
            for label in ("wgrads", "params"):
                relative = "%s/iter_%07d/mp_rank_00.pth" % (label, step)
                values = {arm: self.read_snapshot(arm, relative) for arm in ("native", "optimized")}
                parameter_values[label] = values
                self.native_parameter_dump(values["native"], values["optimized"], initial, metadata, step, label)
                if all(isinstance(value, Mapping) and isinstance(value.get("model_chunk0"), Mapping)
                       for value in values.values()):
                    parameter_keys[label] = {arm: set(value["model_chunk0"]) for arm, value in values.items()}
            if len(parameter_keys) == 2:
                for arm in ("native", "optimized"):
                    self.require(parameter_keys["wgrads"][arm] == parameter_keys["params"][arm],
                                 "%s step %d all gradient and post-SGD parameter keys match" % (arm, step))
            self.master_after(step, metadata, parameter_keys, parameter_values)
        for group in ("initial", "actual_start_model_parameters", "actual_start_master_parameters",
                      "master_initialization", "model_initialization", "batches", "output",
                      "derived_loss", "wgrads", "params", "masters", "master_model_consistency"):
            self.require(self.tensor_groups.get(group, 0) > 0, group + " tensor comparison coverage nonempty")
        passed = all(item["passed"] for item in self.checks)
        return {"schema": "megatron-bf16-formal-entry-audit-comparison-v1", "passed": passed,
                "native_case": str(self.native.resolve()), "optimized_case": str(self.optimized.resolve()),
                "steps": STEPS, "comparison_device": "CPU float64 copies", "atol": ATOL, "rtol": RTOL,
                "strict_equality_scope": "Initial state_dict tensors and every captured input batch tensor; "
                                         "BF16 model and FP32 master optimizer start tensors; all recursively "
                                         "captured non-tensor values must be equivalent.",
                "optimizer_master_scope": "Every FP32 master parameter is compared at initialization and "
                                          "after each of three native SGD updates, independently of rounded "
                                          "BF16 model parameter comparisons.",
                "pre_wrap_initial_dtype_counts": self.initial_dtype_counts,
                "comparator_sha256": sha256(__file__), "torch": str(self.torch.__version__),
                "checks": self.checks, "check_summary": {"total": len(self.checks),
                    "passed": sum(item["passed"] for item in self.checks),
                    "failed": sum(not item["passed"] for item in self.checks)},
                "tensor_groups": self.tensor_groups, "tensor_comparisons": self.tensors,
                "raw_value_comparisons": self.raw_values, "derived_masked_mean_loss": self.derived_losses,
                "adapter_coverage": self.coverage, "input_files": self.files,
                "finished_utc": datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_case", type=Path)
    parser.add_argument("optimized_case", type=Path)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite comparison evidence")
    if args.native_case.resolve() == args.optimized_case.resolve():
        parser.error("native and optimized cases must be separate directories")
    import torch
    comparison = Comparison(args.native_case, args.optimized_case, torch,
                            contract_path=args.contract, candidate_path=args.candidate)
    try:
        report = comparison.run()
    except Exception as error:
        comparison.require(False, "comparison completed without processing errors",
                           error_type=type(error).__name__, error=str(error))
        report = {"schema": "megatron-bf16-formal-entry-audit-comparison-v1", "passed": False,
                  "comparator_sha256": sha256(__file__), "checks": comparison.checks,
                  "tensor_comparisons": comparison.tensors, "input_files": comparison.files,
                  "error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve()),
                      "check_summary": report.get("check_summary")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
