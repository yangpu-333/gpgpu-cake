"""Summarize immutable 120-step native pretrain_gpt.py performance evidence.

This CPU-only reader does not import Megatron, Torch, Triton or candidate code.
It keeps all 12 native timing intervals and averages the 10 ending after step20.
No minimum speedup is imposed: evidence validity and measured gain are separate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import unittest


ROOT = Path(__file__).resolve().parent
COMMIT = "5be9626709af2722333bf54797c954c09edeada3"
CANDIDATE_SHA256 = "4a5c2180114396412b1c99907837ab970b2a9efdffe85e0a46f5527cc0790201"
STEPS = 120
SEEDS = (25002, 25003, 25004)
ARMS = ("native", "optimized")
INTERVAL_ENDS = tuple(range(10, STEPS + 1, 10))
WARMUP_STEPS = 20
TOKENS_PER_STEP = 256
INTEGRATION_FILES = ("launch_pretrain.py", "runtime_compat.py", "kernel_adapter.py",
                     "run_case.py", "contract.json", "source_manifest.json")
MODEL_TYPE = "megatron.core.models.gpt.gpt_model.GPTModel"
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|[+-]?(?:nan|inf(?:inity)?)"


class EvidenceError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def parse_intervals(text):
    """Read complete log-interval10 records and an optional first-step startup.

    Native Megatron additionally logs iteration1 once before iteration10. This
    single startup record is preserved and excluded from interval averaging.
    """
    records = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if "elapsed time per iteration" not in line:
            continue
        iteration = re.search(r"\biteration\s+(\d+)\s*/\s*(\d+)\b", line)
        require(iteration is not None, "timing line %d lacks iteration/total" % line_number)
        index, total = map(int, iteration.groups())
        require(total == STEPS, "timing line %d has total %d, expected120" % (line_number, total))
        require(index == 1 or index in INTERVAL_ENDS, "unexpected timing iteration %d" % index)
        require(index not in seen, "duplicate timing iteration %d" % index)
        if index == 1:
            require(not seen, "startup iteration1 must precede iteration10")
        seen.add(index)
        record = {"iteration": index, "total_iterations": total, "line_number": line_number,
                  "raw_line": line, "retained": index > WARMUP_STEPS,
                  "kind": "startup_first_iteration" if index == 1 else "logged_interval"}
        for key, label in (("mean_iteration_ms", r"elapsed time per iteration\s*\(ms\)"),
                           ("lm_loss", r"lm loss")):
            match = re.search(label + r"\s*:\s*(" + NUMBER + r")\s*(?:\||$)", line, re.IGNORECASE)
            require(match is not None, "timing line %d lacks %s" % (line_number, key))
            value = float(match.group(1))
            require(math.isfinite(value), "timing line %d has nonfinite %s" % (line_number, key))
            if key == "mean_iteration_ms":
                require(value > 0, "timing line %d has nonpositive duration" % line_number)
            record[key] = value
        for key, label, expected in (
                ("consumed_samples", "consumed samples", index * 2),
                ("global_batch_size", "global batch size", 2),
                ("skipped_iterations", "number of skipped iterations", 0),
                ("nan_iterations", "number of nan iterations", 0)):
            match = re.search(re.escape(label) + r"\s*:\s*(\d+)\s*(?:\||$)", line)
            require(match is not None, "timing line %d lacks %s" % (line_number, key))
            record[key] = int(match.group(1))
            require(record[key] == expected,
                    "timing iteration %d %s=%s, expected%s" % (index, key, record[key], expected))
        records.append(record)
    require(tuple(record["iteration"] for record in records if record["iteration"] != 1) == INTERVAL_ENDS,
            "expected ordered timing intervals10,20,...120; observed %r" %
            [record["iteration"] for record in records])
    require(sum(record["retained"] for record in records) == 10, "expected10 retained intervals")
    return records


def normalized_path(value):
    require(isinstance(value, str) and bool(value), "expected nonempty path string")
    return value.replace("\\", "/").rstrip("/")


def same_recorded_path(left, right):
    """Accept different recorded paths only when their actual filesystem agrees.

    A launcher resolves its checkout while run_case retains the user-supplied
    alias. Offline readers cannot establish a remote alias and must reject it.
    """
    left, right = normalized_path(left), normalized_path(right)
    if left == right:
        return True
    a, b = Path(left), Path(right)
    if not a.exists() or not b.exists():
        return False
    return a.resolve(strict=True) == b.resolve(strict=True) and a.samefile(b)


def timestamp(value):
    require(isinstance(value, str), "missing timestamp")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timestamp must include timezone")
    return result


def check_metadata(command, entry, arm, seed, contract, contract_sha, manifest, integration):
    case = "%s-perf-%d" % (arm, seed)
    require(command.get("case") == case and command.get("mode") == arm, "wrong case or arm")
    require(type(command.get("seed")) is int and command["seed"] == seed, "wrong seed")
    require(type(command.get("steps")) is int and command["steps"] == STEPS, "expected120 steps")
    require(command.get("audit") is False, "performance run must disable tensor audit")
    require(command.get("exit_code") == 0 and entry.get("status") == "completed",
            "formal process did not complete successfully")
    require("error" not in command and "error" not in entry, "run recorded an error")
    require(entry.get("exit_code", 0) in (0, None), "launcher exited unsuccessfully")
    require(command.get("contract_sha256") == contract_sha, "wrong command contract SHA256")
    require(entry.get("megatron_commit") == COMMIT, "wrong native commit")
    require(entry.get("native_sources_sha256") == manifest["files"], "native source manifest differs")
    require(entry.get("integration_sha256") == integration, "integration source hashes differ")
    require(entry.get("launcher_sha256") == integration["launch_pretrain.py"], "wrong launcher SHA256")
    expected_training = contract["training_argv"] + ["--train-iters", "120", "--seed", str(seed),
                                                   "--log-interval", "10"]
    require(entry.get("training_argv") == expected_training, "formal training argv differs")
    invocation = command.get("command")
    require(isinstance(invocation, list) and len(invocation) >= 10
            and all(isinstance(value, str) for value in invocation), "missing invocation")
    require(invocation[1] == "-u" and normalized_path(invocation[2]).endswith("/launch_pretrain.py"),
            "wrong invocation launcher")
    require(invocation[3] == "--megatron-root", "missing explicit Megatron root")
    native_root = normalized_path(invocation[4])
    evidence_path = normalized_path(invocation[7])
    require(evidence_path.endswith("/" + case + "/entry.json"), "wrong entry evidence destination")
    invocation_prefix = ["--megatron-root", invocation[4], "--corex42-compat",
                         "--entry-evidence", invocation[7]]
    if arm == "optimized":
        invocation_prefix.append("--iluvatar-kernels")
    require(invocation[3:] == invocation_prefix + expected_training,
            "invocation contains unexpected arguments, audit, evaluation or checkpoint flags")
    require(same_recorded_path(entry.get("formal_entry"), native_root + "/pretrain_gpt.py"),
            "not the formal native pretrain_gpt.py entry; differing aliases require existing same files")
    runtime = entry.get("runtime_compatibility")
    require(isinstance(runtime, dict) and runtime.get("torch") == "2.4.1"
            and runtime.get("optional_backends_disabled_in_process") == ["transformer_engine", "apex"]
            and same_recorded_path(runtime.get("selected_megatron_source"), native_root),
            "wrong common local-PyTorch CoreX runtime")
    require(entry.get("iluvatar_kernels") is (arm == "optimized"), "wrong kernel opt-in state")
    models = entry.get("models")
    require(isinstance(models, list) and len(models) == 1 and isinstance(models[0], dict),
            "expected exactly one model chunk")
    model = models[0]
    for key, expected in {"class": MODEL_TYPE, "parameters": 13701632, "layers": 4,
                          "hidden_size": 512, "vocab_size": 1024}.items():
        require(model.get(key) == expected, "wrong constructed model " + key)
    keys = model.get("state_dict_keys")
    require(isinstance(keys, list) and keys and all(isinstance(key, str) for key in keys)
            and len(keys) == len(set(keys)), "invalid state_dict key inventory")
    adapters = entry.get("kernel_adapters")
    if arm == "native":
        require(adapters == [], "native model has installed adapters")
    else:
        check_adapter(adapters, manifest)
    elapsed = command.get("whole_process_elapsed_seconds")
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0,
            "missing finite whole-process elapsed time")
    command_start, command_end = timestamp(command.get("started_utc")), timestamp(command.get("finished_utc"))
    entry_start, entry_end = timestamp(entry.get("started_utc")), timestamp(entry.get("finished_utc"))
    require(command_start <= entry_start <= entry_end <= command_end, "invalid process chronology")
    return {"model": model, "adapter": adapters[0] if adapters else None,
            "whole_process_elapsed_seconds": elapsed, "started_utc": command["started_utc"],
            "finished_utc": command["finished_utc"], "training_argv": expected_training,
            "formal_entry": entry["formal_entry"], "runtime_compatibility": runtime}


def check_adapter(adapters, manifest):
    require(isinstance(adapters, list) and len(adapters) == 1 and isinstance(adapters[0], dict),
            "optimized requires exactly one model adapter")
    handle = adapters[0]
    require(all(handle.get(key) is True for key in ("enabled", "active", "provenance_verified")),
            "optimized adapter not active and verified")
    for key in ("candidate_sha256", "candidate_expected_sha256"):
        require(handle.get(key) == CANDIDATE_SHA256, "wrong immutable candidate " + key)
    require(handle.get("megatron_commit") == COMMIT and handle.get("model_type") == MODEL_TYPE,
            "wrong adapter model provenance")
    require(handle.get("runtime_world_size") == 1 and
            handle.get("verified_devices") in ({"0": "BI-V150"}, {"0": "Iluvatar BI-V150"}),
            "adapter did not observe BI-V150 world-size1")
    native = handle.get("native_sources_sha256")
    require(isinstance(native, dict) and len(native) == 5 and
            all(manifest["files"].get(key) == value for key, value in native.items()),
            "adapter native source hashes differ")
    cfg = handle.get("config_at_installation", {})
    expected_cfg = {"num_layers": 4, "hidden_size": 512, "num_attention_heads": 8,
                    "ffn_hidden_size": 2048, "num_query_groups": 8, "kv_channels": 64,
                    "normalization": "RMSNorm", "layernorm_epsilon": 1e-5,
                    "hidden_dropout": 0.0, "attention_dropout": 0.0, "add_bias_linear": False,
                    "sequence_parallel": False, "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1, "context_parallel_size": 1,
                    "fp16": False, "bf16": False}
    require(isinstance(cfg, dict), "missing adapter configuration")
    for key, expected in expected_cfg.items():
        require(cfg.get(key) == expected, "wrong adapter config " + key)
    counts = handle.get("counts", {})
    expectations = {
        "residual": {"candidate_api_calls": 480, "cached_norm_calls": 480, "native_bda_calls": 0,
                     "native_norm_calls": 0, "autocast_compatibility_calls": 0, "fallback_reasons": {}},
        "cross_entropy": {"candidate_api_calls": 120, "fp32_triton_eligible_calls": 120,
                          "native_calls": 0, "autocast_torch_compatibility_calls": 0, "fallback_reasons": {}},
    }
    for operation, expected in expectations.items():
        observed = counts.get(operation)
        require(isinstance(observed, dict), "missing " + operation + " counters")
        for key, value in expected.items():
            require(observed.get(key) == value, "incorrect %s/%s coverage" % (operation, key))
        observations = observed.get("observations")
        require(isinstance(observations, list) and len(observations) == 4,
                "expected four recorded " + operation + " shape observations")
        for sample in observations:
            require(isinstance(sample, dict), "shape observation must be an object")
            if operation == "residual":
                for key in ("x", "residual", "normalized"):
                    tensor = sample.get(key, {})
                    require(tensor.get("shape") in ([128, 2, 512], [256, 512])
                            and tensor.get("dtype") == "torch.float32" and tensor.get("device") == "cuda:0",
                            "wrong residual " + key + " shape/dtype/device")
                require(sample.get("epsilon") == 1e-5, "wrong observed RMS epsilon")
            else:
                for key, shape, dtype in (("logits", [128, 2, 1024], "torch.float32"),
                                          ("target", [128, 2], "torch.int64"),
                                          ("loss", [128, 2], "torch.float32")):
                    tensor = sample.get(key, {})
                    require(tensor.get("shape") == shape and tensor.get("dtype") == dtype
                            and tensor.get("device") == "cuda:0",
                            "wrong CE " + key + " shape/dtype/device")


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), "expected JSON object: " + str(path))
    return value


def summarize(evidence_root, contract_path, manifest_path, integration_root, candidate_path):
    contract = read_json(contract_path)
    manifest = read_json(manifest_path)
    contract_sha = sha256(contract_path)
    performance = contract.get("performance", {})
    require(contract.get("megatron_commit") == COMMIT and contract.get("candidate_sha256") == CANDIDATE_SHA256,
            "unexpected formal-entry contract provenance")
    require(performance.get("steps") == STEPS and performance.get("independent_seeds") == list(SEEDS)
            and performance.get("arms") == list(ARMS) and performance.get("tensor_snapshotting") is False,
            "unexpected formal performance design")
    require(manifest.get("commit") == COMMIT and isinstance(manifest.get("files"), dict)
            and len(manifest["files"]) == 600
            and all(isinstance(key, str) and isinstance(value, str) and SHA_PATTERN.fullmatch(value)
                    for key, value in manifest["files"].items()), "unexpected native600-file source manifest")
    require(sha256(candidate_path) == CANDIDATE_SHA256, "candidate file differs from accepted0006")
    integration = {name: sha256(Path(integration_root) / name) for name in INTEGRATION_FILES}
    require(integration["contract.json"] == contract_sha and integration["source_manifest.json"] == sha256(manifest_path),
            "provided contract/manifest differs from integration directory")
    cases, checks, inputs = [], [], []
    for seed in SEEDS:
        for arm in ARMS:
            name = "%s-perf-%d" % (arm, seed)
            folder = Path(evidence_root) / name
            case = {"case": name, "seed": seed, "arm": arm, "passed": False}
            try:
                for filename in ("command.json", "entry.json", "stdout.log", "stderr.log"):
                    path = folder / filename
                    require(path.is_file(), "missing input " + str(path))
                    inputs.append({"case": name, "path": str(path.resolve()), "bytes": path.stat().st_size,
                                   "sha256": sha256(path)})
                command, entry = read_json(folder / "command.json"), read_json(folder / "entry.json")
                case.update(check_metadata(command, entry, arm, seed, contract, contract_sha, manifest, integration))
                require(not (folder / "snapshots").exists(), "performance case contains tensor snapshots")
                intervals = parse_intervals((folder / "stdout.log").read_text(encoding="utf-8"))
                mean = statistics.mean(record["mean_iteration_ms"] for record in intervals if record["retained"])
                complete_intervals = [record for record in intervals if record["kind"] == "logged_interval"]
                startup_records = [record for record in intervals if record["kind"] == "startup_first_iteration"]
                case.update(passed=True, raw_intervals=complete_intervals, startup_records=startup_records,
                            raw_timing_records=intervals, raw_timing_record_count=len(intervals),
                            interval_count=len(complete_intervals),
                            retained_interval_count=10, retained_training_steps=100,
                            arithmetic_mean_iteration_ms=mean, tokens_per_second=1000 * TOKENS_PER_STEP / mean)
                checks.append({"case": name, "passed": True})
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                case.update(error_type=type(error).__name__, error=str(error))
                checks.append({"case": name, "passed": False, "error": str(error)})
            cases.append(case)
    seed_comparisons, process_orders = [], []
    valid = len(cases) == 6 and all(case["passed"] for case in cases)
    if valid:
        try:
            model = cases[0]["model"]
            runtime = cases[0]["runtime_compatibility"]
            entry_path = cases[0]["formal_entry"]
            require(all(case["model"] == model for case in cases), "constructed model metadata differs across runs")
            require(all(case["runtime_compatibility"] == runtime and case["formal_entry"] == entry_path for case in cases),
                    "runtime mappings or native entry path differs across runs")
            for seed in SEEDS:
                pair = {case["arm"]: case for case in cases if case["seed"] == seed}
                ordered = sorted(pair.values(), key=lambda case: timestamp(case["started_utc"]))
                require(timestamp(ordered[0]["finished_utc"]) <= timestamp(ordered[1]["started_utc"]),
                        "seed%d processes overlap" % seed)
                process_orders.append({"seed": seed, "first": ordered[0]["arm"], "second": ordered[1]["arm"]})
                ratio = pair["native"]["arithmetic_mean_iteration_ms"] / pair["optimized"]["arithmetic_mean_iteration_ms"]
                seed_comparisons.append({"seed": seed, "native_mean_ms": pair["native"]["arithmetic_mean_iteration_ms"],
                                         "optimized_mean_ms": pair["optimized"]["arithmetic_mean_iteration_ms"],
                                         "native_tokens_per_second": pair["native"]["tokens_per_second"],
                                         "optimized_tokens_per_second": pair["optimized"]["tokens_per_second"],
                                         "native_over_optimized_ratio": ratio,
                                         "throughput_change_percent": (ratio - 1) * 100})
            require(all(left["first"] != right["first"] for left, right in zip(process_orders, process_orders[1:])),
                    "process order does not alternate across seeds")
            chronological = sorted(cases, key=lambda case: timestamp(case["started_utc"]))
            require(all(timestamp(left["finished_utc"]) <= timestamp(right["started_utc"])
                        for left, right in zip(chronological, chronological[1:])), "performance processes overlap")
            checks.append({"name": "common model/runtime and sequential alternating process order", "passed": True})
        except (ValueError, TypeError, KeyError) as error:
            checks.append({"name": "common model/runtime and sequential alternating process order", "passed": False,
                           "error": str(error)})
            valid = False
    geomean = (math.exp(statistics.mean(math.log(item["native_over_optimized_ratio"]) for item in seed_comparisons))
               if valid and len(seed_comparisons) == len(SEEDS) else None)
    return {
        "schema": "megatron-formal-entry-performance-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": valid, "summarizer_sha256": sha256(__file__), "contract_sha256": contract_sha,
        "candidate_sha256": CANDIDATE_SHA256, "megatron_commit": COMMIT,
        "native_sources_sha256": manifest["files"], "integration_sha256": integration,
        "timing": {"interface": "native formal-loop GPU-synchronized time.time interval timer reported in stdout",
                   "log_interval_steps": 10, "all_interval_count_per_arm": 12,
                   "optional_first_iteration_startup_record": "at most one iteration1 record, only before iteration10; excluded",
                   "excluded_iteration_ends": [10, 20], "retained_interval_count_per_arm": 10,
                   "tokens_per_step": TOKENS_PER_STEP, "tokens_per_second_formula": "256000 / mean_iteration_ms",
                   "mean": "arithmetic mean of ten equal-length logged intervals ending30..120",
                   "includes": "formal-loop data loading, scheduling, DDP, forward/backward, SGD",
                   "excludes": "startup, first20training iterations, evaluation and checkpointing",
                   "whole_process_elapsed_scope": "includes startup and shutdown; kept separately from throughput"},
        "cases": cases, "seed_comparisons": seed_comparisons, "process_order": process_orders,
        "geometric_mean_native_over_optimized_ratio": geomean,
        "throughput_change_percent": (geomean - 1) * 100 if geomean is not None else None,
        "measured_gain": valid and geomean is not None and geomean > 1,
        "all_seeds_show_gain": valid and all(item["native_over_optimized_ratio"] > 1 for item in seed_comparisons),
        "minimum_speedup_acceptance_threshold": None,
        "checks": checks, "input_files": inputs,
        "limits": ["Fixed FP32 small4-layer GPT, native MockGPTDataset/NullTokenizer, TP/PP/CP/world1 on BI-V150.",
                   "This is formal-entry interval throughput, distinct from the historical19.10% benchmark result.",
                   "Same-seed arm results come from separate processes; three independent seeds, no confidence claim.",
                   "Durations are rounded native stdout measurements; retained raw lines preserve their precision.",
                   "Adapter counts establish API dispatch and no fallback, not an independent GPU launch trace.",
                   "Performance runs do not capture tensors; correctness must be reported from the separate3-step audit."]}


class ParserTests(unittest.TestCase):
    @staticmethod
    def valid_log():
        return "\n".join(
            " [2026-10-03 12:00:55.388732] iteration %8d/ %7d | consumed samples: %12d | "
            "elapsed time per iteration (ms): %.1f | learning rate: 1.000000E-03 | "
            "global batch size: 2 | lm loss: 7.062391E+00 | loss scale: 1.0 | grad norm: 7.540 | "
            "number of skipped iterations: 0 | number of nan iterations: 0 |" % (i, STEPS, i * 2, 100 + i)
            for i in INTERVAL_ENDS)

    def test_valid_intervals_and_warmup(self):
        intervals = parse_intervals(self.valid_log())
        self.assertEqual(len(intervals), 12)
        self.assertEqual(sum(item["retained"] for item in intervals), 10)
        self.assertEqual(statistics.mean(item["mean_iteration_ms"] for item in intervals if item["retained"]), 175)

    def test_duplicate_interval_fails(self):
        lines = self.valid_log().splitlines()
        with self.assertRaisesRegex(EvidenceError, "duplicate"):
            parse_intervals("\n".join(lines + [lines[-1]]))

    @classmethod
    def log_with_startup(cls):
        log = cls.valid_log()
        first = log.splitlines()[0]
        startup = re.sub(r"iteration\s+10/", "iteration        1/", first)
        startup = re.sub(r"consumed samples:\s+20", "consumed samples:            2", startup)
        startup = startup.replace("(ms): 110.0", "(ms): 570.0")
        return startup + "\n" + log

    def test_native_first_iteration_startup_is_preserved_and_excluded(self):
        records = parse_intervals(self.log_with_startup())
        self.assertEqual(len(records), 13)
        self.assertEqual(records[0]["iteration"], 1)
        self.assertEqual(records[0]["kind"], "startup_first_iteration")
        self.assertFalse(records[0]["retained"])
        self.assertEqual(sum(item["kind"] == "logged_interval" for item in records), 12)
        self.assertEqual(sum(item["retained"] for item in records), 10)
        self.assertEqual(statistics.mean(item["mean_iteration_ms"] for item in records if item["retained"]), 175)

    def test_duplicate_late_or_gap_with_startup_fails(self):
        lines = self.log_with_startup().splitlines()
        for changed in ([lines[0]] + lines, [lines[1], lines[0]] + lines[2:], lines[:2] + lines[3:]):
            with self.subTest(first_lines=changed[:2]), self.assertRaises(EvidenceError):
                parse_intervals("\n".join(changed))

    def test_missing_interval_fails(self):
        with self.assertRaisesRegex(EvidenceError, "expected ordered"):
            parse_intervals("\n".join(self.valid_log().splitlines()[1:]))

    def test_wrong_total_fails(self):
        with self.assertRaisesRegex(EvidenceError, "total 119"):
            parse_intervals(self.valid_log().replace("/     120", "/     119"))

    def test_skipped_nan_and_consumed_counts_fail(self):
        for original, changed in (("skipped iterations: 0", "skipped iterations: 1"),
                                  ("nan iterations: 0", "nan iterations: 1"),
                                  ("consumed samples:           20", "consumed samples:           19")):
            with self.subTest(changed=changed), self.assertRaises(EvidenceError):
                parse_intervals(self.valid_log().replace(original, changed, 1))

    def test_nonfinite_loss_or_nonpositive_time_fails(self):
        for original, changed in (("lm loss: 7.062391E+00", "lm loss: nan"),
                                  ("(ms): 110.0", "(ms): 0.0")):
            with self.subTest(changed=changed), self.assertRaises(EvidenceError):
                parse_intervals(self.valid_log().replace(original, changed, 1))

    def test_out_of_order_interval_fails(self):
        lines = self.valid_log().splitlines()
        lines[2], lines[3] = lines[3], lines[2]
        with self.assertRaisesRegex(EvidenceError, "expected ordered"):
            parse_intervals("\n".join(lines))

    def test_path_alias_requires_actual_same_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder / "pretrain_gpt.py"
            source.write_text("# fixed native entry\n", encoding="utf-8")
            canonical_alias = folder / "child" / ".." / "pretrain_gpt.py"
            (folder / "child").mkdir()
            self.assertTrue(same_recorded_path(str(source), str(canonical_alias)))
            different = folder / "different.py"
            different.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
            self.assertFalse(same_recorded_path(str(source), str(different)))
            self.assertFalse(same_recorded_path(str(source), str(folder / "offline-alias.py")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_root", nargs="?", type=Path)
    parser.add_argument("output", nargs="?", type=Path)
    parser.add_argument("--contract", type=Path, default=ROOT / "contract.json")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "source_manifest.json")
    parser.add_argument("--integration-root", type=Path, default=ROOT)
    parser.add_argument("--candidate", type=Path, default=ROOT.parent / "megatron_cc/higher_gain/candidates/0006/candidate.py")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ParserTests))
        return 0 if result.wasSuccessful() else 1
    if args.evidence_root is None or args.output is None:
        parser.error("evidence_root and output are required unless --self-test")
    if args.output.exists():
        parser.error("refusing to overwrite performance evidence")
    try:
        report = summarize(args.evidence_root, args.contract, args.source_manifest, args.integration_root, args.candidate)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        report = {"schema": "megatron-formal-entry-performance-v1", "passed": False,
                  "summarizer_sha256": sha256(__file__), "measured_gain": False,
                  "error_type": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], "output": str(args.output.resolve()),
                      "geometric_mean_native_over_optimized_ratio": report.get("geometric_mean_native_over_optimized_ratio"),
                      "measured_gain": report["measured_gain"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
