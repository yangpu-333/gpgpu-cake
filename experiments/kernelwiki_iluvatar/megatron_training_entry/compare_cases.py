"""Compare two immutable three-step formal Megatron audit cases on CPU.

This reads only tensor snapshots emitted by launch_pretrain.py and the pinned
native save_grads routine. It never imports Megatron, candidate kernels, or
training code. Full training checkpoints are inventoried but not unpickled.
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
CANDIDATE_SHA256 = "4a5c2180114396412b1c99907837ab970b2a9efdffe85e0a46f5527cc0790201"
ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Comparison:
    def __init__(self, native, optimized, torch_module):
        self.native, self.optimized = Path(native), Path(optimized)
        self.torch = torch_module
        self.checks, self.tensors, self.raw_values, self.files = [], [], [], []
        self.tensor_groups = {}
        self.coverage = {}
        self.derived_losses = []

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
        contract_path, manifest_path = ROOT / "contract.json", ROOT / "source_manifest.json"
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.inventory(contract_path, "fixed-verifier")
        self.inventory(manifest_path, "fixed-verifier")
        correctness = contract.get("correctness", {})
        self.require(correctness.get("atol") == ATOL and correctness.get("rtol") == RTOL
                     and correctness.get("steps") == STEPS,
                     "fixed contract supplies the exact comparison tolerances and three steps")
        self.require(contract.get("megatron_commit") == COMMIT
                     and contract.get("candidate_sha256") == CANDIDATE_SHA256,
                     "fixed contract pins native commit and accepted candidate")
        self.require(manifest.get("commit") == COMMIT and isinstance(manifest.get("files"), dict)
                     and bool(manifest.get("files")), "fixed native source manifest present")
        records = {arm: {name: self.read_json(arm, name) for name in ("command.json", "entry.json")}
                   for arm in ("native", "optimized")}
        commands = [records[arm]["command.json"] for arm in ("native", "optimized")]
        entries = [records[arm]["entry.json"] for arm in ("native", "optimized")]
        if any(value is None for value in commands + entries):
            return None
        for arm, command, entry in zip(("native", "optimized"), commands, entries):
            self.require(command.get("mode") == arm, arm + " run mode")
            self.require(command.get("audit") is True and command.get("steps") == STEPS,
                         arm + " is a three-step tensor audit")
            self.require(command.get("exit_code") == 0 and entry.get("status") == "completed",
                         arm + " formal launch completed")
            self.require(entry.get("megatron_commit") == COMMIT, arm + " pinned Megatron commit")
            self.require(isinstance(entry.get("models"), list) and len(entry["models"]) == 1,
                         arm + " exactly model chunk 0")
            self.require(bool(command.get("contract_sha256")), arm + " contract hash present")
        self.require(commands[0].get("seed") == commands[1].get("seed")
                     and isinstance(commands[0].get("seed"), int), "same explicit seed")
        self.require(commands[0].get("contract_sha256") == commands[1].get("contract_sha256"),
                     "same contract SHA256")
        integration_names = {"launch_pretrain.py", "runtime_compat.py", "kernel_adapter.py", "run_case.py",
                             "contract.json", "source_manifest.json"}
        expected_integration = {name: sha256(ROOT / name) for name in integration_names}
        for arm, command, entry in zip(("native", "optimized"), commands, entries):
            self.require(command.get("contract_sha256") == expected_integration["contract.json"],
                         arm + " command matches the fixed verifier contract SHA256")
            self.require(entry.get("native_sources_sha256") == manifest.get("files"),
                         arm + " complete native source hashes match fixed manifest",
                         expected_files=len(manifest.get("files", {})),
                         observed_files=len(entry.get("native_sources_sha256", {})))
            self.require(entry.get("integration_sha256") == expected_integration,
                         arm + " complete integration hashes match verifier source files")
            self.require(entry.get("launcher_sha256") == expected_integration["launch_pretrain.py"],
                         arm + " final launcher SHA256 remains frozen")
        self.require(entries[0].get("native_sources_sha256") == entries[1].get("native_sources_sha256"),
                     "both arms have identical complete native source hashes")
        self.require(entries[0].get("integration_sha256") == entries[1].get("integration_sha256"),
                     "both arms have identical integration hashes")
        self.compare_tree(entries[0].get("models"), entries[1].get("models"),
                          "model construction metadata", "metadata", exact=True)
        for arm, entry in zip(("native", "optimized"), entries):
            self.require(isinstance(entry.get("training_argv"), list), arm + " training argv present")
        if all(isinstance(entry.get("training_argv"), list) for entry in entries):
            self.require(self.normalized_argv(entries[0]["training_argv"]) ==
                         self.normalized_argv(entries[1]["training_argv"]),
                         "identical formal training argv except output directory")
        self.require(entries[0].get("iluvatar_kernels") is False
                     and entries[0].get("kernel_adapters") == [], "native has no candidate installation")
        self.require(entries[1].get("iluvatar_kernels") is True, "optimized explicitly opted in")
        adapters = entries[1].get("kernel_adapters")
        if not self.require(isinstance(adapters, list) and len(adapters) == 1,
                            "optimized has exactly one model adapter"):
            return records
        handle = adapters[0]
        if not self.require(isinstance(handle, dict), "adapter metadata is an object"):
            return records
        self.require(all(handle.get(key) is True for key in ("enabled", "active", "provenance_verified")),
                     "optimized adapter active and provenance verified")
        self.require(handle.get("candidate_sha256") == CANDIDATE_SHA256 and
                     handle.get("candidate_expected_sha256") == CANDIDATE_SHA256,
                     "exact accepted candidate SHA256")
        self.require(handle.get("megatron_commit") == COMMIT, "adapter pinned Megatron commit")
        self.require(handle.get("runtime_world_size") == 1, "adapter observed world size one")
        devices = handle.get("verified_devices", {})
        self.require(isinstance(devices, dict) and set(devices) == {"0"} and
                     devices.get("0") in ("BI-V150", "Iluvatar BI-V150"), "adapter observed BI-V150 device zero")
        counts = handle.get("counts", {})
        residual, ce = counts.get("residual", {}), counts.get("cross_entropy", {})
        expected_residual = 4 * STEPS
        expectations = {
            "residual": {"candidate_api_calls": expected_residual, "cached_norm_calls": expected_residual,
                         "native_bda_calls": 0, "native_norm_calls": 0, "autocast_compatibility_calls": 0,
                         "fallback_reasons": {}},
            "cross_entropy": {"candidate_api_calls": STEPS, "fp32_triton_eligible_calls": STEPS,
                              "native_calls": 0, "autocast_torch_compatibility_calls": 0, "fallback_reasons": {}},
        }
        for operation, observed in (("residual", residual), ("cross_entropy", ce)):
            for key, expected in expectations[operation].items():
                self.require(isinstance(observed, dict) and observed.get(key) == expected,
                             "adapter %s/%s coverage" % (operation, key), expected=expected,
                             observed=observed.get(key) if isinstance(observed, dict) else None)
        cfg = handle.get("config_at_installation", {})
        for name, expected in {"num_layers": 4, "hidden_size": 512, "num_attention_heads": 8,
                               "ffn_hidden_size": 2048, "normalization": "RMSNorm",
                               "layernorm_epsilon": 1e-5, "attention_dropout": 0.0,
                               "hidden_dropout": 0.0, "add_bias_linear": False, "sequence_parallel": False,
                               "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
                               "context_parallel_size": 1, "fp16": False, "bf16": False}.items():
            self.require(isinstance(cfg, dict) and cfg.get(name) == expected,
                         "adapter fixed configuration " + name)
        self.coverage = {"expected": expectations, "observed": counts,
                         "scope": "Aggregate API dispatch and cached norm use across four layers and three forwards. "
                                  "These counters do not independently prove GPU kernel launches."}
        return records

    def snapshot_layout(self):
        expected = {"model0-initial.pt"} | {"model0-forward-%03d.pt" % step for step in range(1, STEPS + 1)}
        for arm, directory in (("native", self.native), ("optimized", self.optimized)):
            snapshots = directory / "snapshots"
            observed = {path.name for path in snapshots.glob("model*-*.pt")}
            self.require(observed == expected, arm + " exact initial/forward snapshot inventory",
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

    def native_parameter_dump(self, native, optimized, initial, metadata, step, label):
        expected_keys = {"model_chunk0"}
        valid = all(isinstance(value, Mapping) and set(value) == expected_keys for value in (native, optimized))
        if not self.require(valid, "step %d/%s exactly model_chunk0" % (step, label)):
            return
        chunks = [value["model_chunk0"] for value in (native, optimized)]
        if not self.require(all(isinstance(value, Mapping) and bool(value) for value in chunks),
                            "step %d/%s nonempty parameter mapping" % (step, label)):
            return
        for arm, chunk in zip(("native", "optimized"), chunks):
            all_tensors = all(self.torch.is_tensor(value) for value in chunk.values())
            self.require(all_tensors, "%s step %d/%s contains only native parameter tensors" % (arm, step, label))
            self.require(all_tensors and all(value.dtype == self.torch.float32 for value in chunk.values()),
                         "%s step %d/%s all saved parameter tensors are FP32" % (arm, step, label))
            if all_tensors and metadata:
                entry = metadata[arm]["entry.json"]
                models = entry.get("models", [])
                expected_numel = models[0].get("parameters") if len(models) == 1 else None
                self.require(sum(value.numel() for value in chunk.values()) == expected_numel,
                             "%s step %d/%s covers every recorded parameter element" % (arm, step, label),
                             expected_numel=expected_numel, observed_numel=sum(value.numel() for value in chunk.values()))
            if isinstance(initial[arm], Mapping):
                self.require(set(chunk) <= set(initial[arm]),
                             "%s step %d/%s parameter names occur in initial state_dict" % (arm, step, label))
        self.compare_tree(native, optimized, "step %d/%s" % (step, label), label)

    def run(self):
        metadata = self.metadata()
        self.snapshot_layout()
        initial = {arm: self.read_snapshot(arm, "model0-initial.pt") for arm in ("native", "optimized")}
        self.require(all(isinstance(value, Mapping) and bool(value) for value in initial.values()),
                     "both initial state_dict snapshots are nonempty mappings")
        if all(value is not None for value in initial.values()):
            self.compare_tree(initial["native"], initial["optimized"], "initial state_dict", "initial", exact=True)
            if metadata:
                for arm in ("native", "optimized"):
                    models = metadata[arm]["entry.json"].get("models", [])
                    if len(models) == 1 and isinstance(initial[arm], Mapping):
                        self.require(list(initial[arm]) == models[0].get("state_dict_keys"),
                                     arm + " initial state_dict keys match builder record")
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
                                 tuple(value["output"].shape) == (2, 128) for value in forward.values()),
                             "step %d complete FP32 per-token model loss shape" % step)
                self.compare_tree(forward["native"].get("output"), forward["optimized"].get("output"),
                                  "step %d/output" % step, "output")
                self.derived_loss(forward["native"], forward["optimized"], step)
            else:
                self.require(False, "step %d both forward snapshots are mappings" % step)
            parameter_keys = {}
            for label in ("wgrads", "params"):
                relative = "%s/iter_%07d/mp_rank_00.pth" % (label, step)
                values = {arm: self.read_snapshot(arm, relative) for arm in ("native", "optimized")}
                self.native_parameter_dump(values["native"], values["optimized"], initial, metadata, step, label)
                if all(isinstance(value, Mapping) and isinstance(value.get("model_chunk0"), Mapping)
                       for value in values.values()):
                    parameter_keys[label] = {arm: set(value["model_chunk0"]) for arm, value in values.items()}
            if len(parameter_keys) == 2:
                for arm in ("native", "optimized"):
                    self.require(parameter_keys["wgrads"][arm] == parameter_keys["params"][arm],
                                 "%s step %d all gradient and post-SGD parameter keys match" % (arm, step))
        for group in ("initial", "batches", "output", "wgrads", "params"):
            self.require(self.tensor_groups.get(group, 0) > 0, group + " tensor comparison coverage nonempty")
        passed = all(item["passed"] for item in self.checks)
        return {"schema": "megatron-formal-entry-audit-comparison-v1", "passed": passed,
                "native_case": str(self.native.resolve()), "optimized_case": str(self.optimized.resolve()),
                "steps": STEPS, "comparison_device": "CPU float64 copies", "atol": ATOL, "rtol": RTOL,
                "strict_equality_scope": "Initial state_dict tensors and every captured input batch tensor; "
                                         "all recursively captured non-tensor values must be equivalent.",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite comparison evidence")
    if args.native_case.resolve() == args.optimized_case.resolve():
        parser.error("native and optimized cases must be separate directories")
    import torch
    comparison = Comparison(args.native_case, args.optimized_case, torch)
    try:
        report = comparison.run()
    except Exception as error:
        comparison.require(False, "comparison completed without processing errors",
                           error_type=type(error).__name__, error=str(error))
        report = {"schema": "megatron-formal-entry-audit-comparison-v1", "passed": False,
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
