"""Apply frozen native and normalized-control gates to the extended final runs.

The candidate and current-best controls run in independent processes. Their
speedup ratio normalizes each arm to its own native measurement; it is not a
same-process paired, three-arm comparison. Raw evidence is never modified.
"""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


BASE_HARNESS_SHA256 = "360e57d1d1baa9d2ddc477224576a7ce87864ff9c3e0aefde546fbf4266cc96a"
PARENT_CONTRACT_SHA256 = "13f71ba192ea8fdf474e89729d2c19fa1a17e9aee3b125a7966abca83941a883"
BEST_CANDIDATE_SHA256 = "b426ef0a309d41450e00e4ce2a1e32a71567a75b08776e27fa6fbc3bb4c566a8"
FULL_CE_CASES = 630
FINAL_MODEL_STEPS = 128
PRIMARY_SEEDS = (24002, 24003, 24004)
EXPECTED = {f"{arm}-{seed}": (arm, seed)
            for arm in ("residual", "ce", "primary", "best")
            for seed in PRIMARY_SEEDS}
EXPECTED.update({"holdout-24005": ("holdout", 24005),
                 "autocast-24006": ("autocast", 24006)})


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def geomean(values):
    if not values or not all(finite_number(v, positive=True) for v in values):
        return None
    return math.exp(statistics.mean(math.log(v) for v in values))


def summary_valid(summary, residual=False):
    if not isinstance(summary, dict):
        return False
    keys = ("finite_native_cases", "finite_correctness_passed",
            "native_invalid_cases", "native_invalid_compatibility_passed")
    if not all(isinstance(summary.get(k), int) and not isinstance(summary[k], bool)
               and summary[k] >= 0 for k in keys):
        return False
    finite, passed, invalid, compatible = (summary[k] for k in keys)
    if finite <= 0 or finite != passed or invalid != compatible:
        return False
    if residual:
        return finite == 224 and invalid == 16
    observed, expected = summary.get("observed_cases"), summary.get("expected_cases")
    return (isinstance(expected, int) and not isinstance(expected, bool)
            and expected == FULL_CE_CASES and observed == expected == finite + invalid)


def training_valid(training, timing):
    if not isinstance(training, dict):
        return False
    medians = training.get("median_ms", {})
    if not all(finite_number(medians.get(k), positive=True) for k in ("native", "candidate")):
        return False
    speedup = training.get("speedup")
    expected_ratio = medians["native"] / medians["candidate"]
    if not finite_number(speedup, positive=True) or not math.isclose(
            speedup, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
        return False
    for field, contract_field in (("warmup", "training_warmup_steps"),
                                  ("samples", "training_samples"),
                                  ("steps_per_sample", "steps_per_sample")):
        if contract_field not in timing or training.get(field) != timing[contract_field]:
            return False
    samples = timing["training_samples"]
    for mapping, positive in (("raw_ms", True), ("incremental_peak_bytes", False)):
        values = training.get(mapping, {})
        if not all(isinstance(values.get(arm), list) and len(values[arm]) == samples
                   and all(finite_number(v, positive=positive) for v in values[arm])
                   for arm in ("native", "candidate")):
            return False
    return all(math.isclose(statistics.median(training["raw_ms"][arm]), medians[arm],
                            rel_tol=1e-12, abs_tol=1e-12)
               for arm in ("native", "candidate"))


def memory_fraction(training):
    peaks = training.get("incremental_peak_bytes", {})
    if not all(isinstance(peaks.get(k), list) and peaks[k]
               and all(finite_number(v) for v in peaks[k]) for k in ("native", "candidate")):
        return None
    native, candidate = (statistics.median(peaks[k]) for k in ("native", "candidate"))
    return candidate / native - 1 if native > 0 else (0.0 if candidate == 0 else None)


def model_identity_valid(model, arm, contract):
    if not isinstance(model, dict) or not isinstance(model.get("configuration"), dict):
        return False
    configuration = model["configuration"]
    expected = contract["holdout_model"] if arm == "holdout" else contract["model"]
    mapping = {"layers": "num_layers", "hidden": "hidden_size", "heads": "num_attention_heads",
               "sequence": "sequence_length", "micro_batch": "micro_batch_size", "vocab": "vocab_size"}
    return (all(configuration.get(key) == expected.get(contract_key) for key, contract_key in mapping.items())
            and configuration.get("autocast") is (arm == "autocast"))


def model_coverage_valid(model, arm, contract):
    if not isinstance(model, dict) or not isinstance(model.get("candidate_calls"), dict):
        return False
    calls = model["candidate_calls"]
    expected = contract["holdout_model"] if arm == "holdout" else contract["model"]
    if calls.get("fused") != expected["num_layers"] * FINAL_MODEL_STEPS or calls.get("fallback") != 0:
        return False
    ce = calls.get("cross_entropy")
    if arm == "best":
        return ce is None
    return (isinstance(ce, dict) and ce.get("optimized") == FINAL_MODEL_STEPS
            and ce.get("fallback") == 0 and not ce.get("fallback_reasons"))


def dispatch_audit_valid(audit, hashes, contract):
    if not isinstance(audit, dict):
        return False
    expected_hashes = {k: hashes[k] for k in ("candidate_sha256", "base_harness_sha256",
                                             "extension_harness_sha256", "contract_sha256")}
    expected_hashes["base_contract_sha256"] = hashes["parent_contract_sha256"]
    shape = [contract["model"][k] for k in ("sequence_length", "micro_batch_size", "vocab_size")]
    if not (all(audit.get(k) == v for k, v in expected_hashes.items())
            and audit.get("passed") is True and audit.get("operation") == "cross_entropy"
            and audit.get("shape") == shape and audit.get("logits_dtype") == "torch.float32"
            and audit.get("loss_dtype") == "torch.float32"):
        return False
    for phase in ("forward", "backward"):
        inventory = audit.get(phase)
        if not isinstance(inventory, dict):
            return False
        compiled, launches = inventory.get("compiled_kernel_names"), inventory.get("launch_counts")
        if not (isinstance(compiled, list) and compiled
                and all(isinstance(name, str) and name.strip() for name in compiled)
                and len(compiled) == len(set(compiled)) and isinstance(launches, dict) and launches
                and set(launches).issubset(compiled)
                and all(isinstance(count, int) and not isinstance(count, bool) and count > 0
                        for count in launches.values())):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id")
    parser.add_argument("final_dir", type=Path)
    parser.add_argument("output", type=Path)
    for name in ("contract", "candidate", "best", "base", "extension", "parent-contract", "dispatch-audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite immutable decision")
    contract = json.loads(args.contract.read_text(encoding="utf-8-sig"))
    parent_contract = json.loads(args.parent_contract.read_text(encoding="utf-8-sig"))
    gates = contract["promotion"]
    timing = contract.get("timing", parent_contract["timing"])
    native_min = gates["primary_wall_speedup_geomean_min"]
    control_min = gates["current_best_normalized_speedup_geomean_min"]
    robustness_min = gates["holdout_and_autocast_speedup_min"]
    memory_max = gates["memory_regression_max_fraction"]
    thresholds_valid = (finite_number(native_min, positive=True) and native_min >= 1.05
                        and finite_number(control_min, positive=True) and control_min >= 1.02
                        and finite_number(robustness_min, positive=True) and robustness_min >= 0.98
                        and finite_number(memory_max) and memory_max <= 0.05)
    hashes = {"candidate_sha256": sha(args.candidate), "best_candidate_sha256": sha(args.best),
              "base_harness_sha256": sha(args.base), "extension_harness_sha256": sha(args.extension),
              "contract_sha256": sha(args.contract), "parent_contract_sha256": sha(args.parent_contract)}
    fixed_sources = (hashes["base_harness_sha256"] == BASE_HARNESS_SHA256
                     and hashes["parent_contract_sha256"] == PARENT_CONTRACT_SHA256
                     and hashes["best_candidate_sha256"] == BEST_CANDIDATE_SHA256
                     and contract.get("parent_contract_sha256") == hashes["parent_contract_sha256"]
                     and contract.get("frozen_base_harness_sha256") == hashes["base_harness_sha256"]
                     and contract.get("current_best_candidate_sha256") == hashes["best_candidate_sha256"])
    fixed_timing = (all(timing.get(k) == parent_contract["timing"].get(k) for k in
                        ("training_warmup_steps", "training_samples", "steps_per_sample", "independent_processes"))
                    and 3 + timing["training_warmup_steps"] + timing["training_samples"]
                    * timing["steps_per_sample"] == FINAL_MODEL_STEPS)
    issues, reports = [], {}
    audit = None
    try:
        audit = json.loads(args.dispatch_audit.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError) as exc:
        issues.append({"run": "dispatch-audit", "error": "unreadable required dispatch audit", "detail": str(exc)})
    audited = dispatch_audit_valid(audit, hashes, contract)
    if not audited:
        issues.append({"run": "dispatch-audit", "error": "actual CE kernel execution gate failed"})
    for name in EXPECTED:
        path = args.final_dir / (name + ".json")
        if not path.is_file():
            issues.append({"run": name, "error": "missing report"})
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(report, dict):
                raise ValueError("report is not a JSON object")
            reports[name] = report
        except (ValueError, UnicodeError) as exc:
            issues.append({"run": name, "error": "unreadable report", "detail": str(exc)})
    complete = set(reports) == set(EXPECTED)
    identity, sources, correctness, measurements, coverage = {}, {}, {}, {}, {}
    for name, (arm, seed) in EXPECTED.items():
        report = reports.get(name, {})
        mode = "operator" if arm == "residual" else ("ce" if arm == "ce" else "model")
        identity[name] = report.get("seed") == seed and report.get("mode") == mode
        if arm != "residual":
            identity[name] = identity[name] and report.get("variant") == ("current-best" if arm == "best" else "extended")
        if mode == "model":
            identity[name] = identity[name] and model_identity_valid(report.get("model"), arm, contract)
        expected_hashes = {"candidate_sha256": hashes["best_candidate_sha256"]
                           if arm == "best" else hashes["candidate_sha256"]}
        if arm == "residual":
            expected_hashes.update(harness_sha256=hashes["base_harness_sha256"],
                                   contract_sha256=hashes["parent_contract_sha256"])
        else:
            expected_hashes.update({k: hashes[k] for k in
                                   ("base_harness_sha256", "extension_harness_sha256", "contract_sha256")})
            expected_hashes["base_contract_sha256"] = hashes["parent_contract_sha256"]
        sources[name] = all(report.get(k) == v for k, v in expected_hashes.items())
        status_passed = report.get("status") == "passed"
        if arm in ("residual", "ce"):
            correctness[name] = status_passed and summary_valid(report.get("case_summary"), arm == "residual")
            measurements[name] = True
            coverage[name] = True
        else:
            model = report.get("model", {})
            correctness[name] = status_passed and isinstance(model, dict) and model.get("passed") is True
            training = model.get("training") if isinstance(model, dict) else None
            measurements[name] = training_valid(training, timing)
            coverage[name] = model_coverage_valid(model, arm, contract)
        for check_name, per_run in (("identity", identity), ("source hashes", sources),
                                    ("correctness", correctness), ("measurement", measurements),
                                    ("adapter coverage", coverage)):
            if not per_run[name]:
                issues.append({"run": name, "error": check_name + " gate failed"})

    trainings = {name: report["model"]["training"] for name, report in reports.items()
                 if name.startswith(("primary-", "best-", "holdout-", "autocast-"))
                 and measurements.get(name) and isinstance(report.get("model"), dict)}
    primary = [trainings[f"primary-{seed}"]["speedup"] for seed in PRIMARY_SEEDS
               if f"primary-{seed}" in trainings]
    best = [trainings[f"best-{seed}"]["speedup"] for seed in PRIMARY_SEEDS
            if f"best-{seed}" in trainings]
    normalized = [trainings[f"primary-{seed}"]["speedup"] / trainings[f"best-{seed}"]["speedup"]
                  for seed in PRIMARY_SEEDS
                  if f"primary-{seed}" in trainings and f"best-{seed}" in trainings]
    primary_geomean, normalized_geomean = geomean(primary), geomean(normalized)
    robustness = {name: trainings[name]["speedup"] for name in ("holdout-24005", "autocast-24006")
                  if name in trainings}
    memory = {name: memory_fraction(training) for name, training in trainings.items()}
    checks = {
        "all_14_required_runs_complete": complete,
        "run_modes_and_seeds_match_manifest": all(identity.values()),
        "frozen_parent_and_current_best_sources_match": fixed_sources,
        "all_recorded_source_hashes_match_exact_files": all(sources.values()),
        "all_correctness_and_native_compatibility_pass": all(correctness.values()),
        "all_training_measurements_match_frozen_protocol": all(measurements.values()),
        "contract_training_protocol_matches_frozen_parent": fixed_timing,
        "all_model_adapter_counts_cover_every_step": all(coverage.values()),
        "main_shape_ce_forward_and_backward_kernel_execution_audited": audited,
        "contract_thresholds_are_at_least_required_strictness": thresholds_valid,
        "three_primary_rounds_each_faster_than_native": len(primary) == 3 and all(v > 1 for v in primary),
        "native_primary_geomean_meets_contract": primary_geomean is not None and primary_geomean >= native_min,
        "three_normalized_control_comparisons_each_faster": len(normalized) == 3 and all(v > 1 for v in normalized),
        "normalized_control_geomean_meets_contract": normalized_geomean is not None and normalized_geomean >= control_min,
        "holdout_and_autocast_meet_contract": len(robustness) == 2 and all(v >= robustness_min for v in robustness.values()),
        "all_eight_training_memory_regressions_meet_contract": len(memory) == 8 and all(
            v is not None and v <= memory_max for v in memory.values()),
    }
    result = {
        "candidate_id": args.candidate_id, "contract": contract.get("version"), **hashes,
        "decision": "promote_for_tested_scope" if all(checks.values()) else "reject_or_experimental",
        "checks": checks, "thresholds": {"native_geomean_min": native_min,
            "normalized_current_best_geomean_min": control_min,
            "holdout_and_autocast_min": robustness_min, "memory_regression_max_fraction": memory_max},
        "primary_speedups_vs_native": primary, "primary_geomean_vs_native": primary_geomean,
        "current_best_speedups_vs_own_native": best,
        "normalized_speedups_vs_current_best": normalized,
        "normalized_geomean_vs_current_best": normalized_geomean,
        "control_comparison_method": "Normalized independent-process comparisons: (extension/native) / (current-best/native). These are not same-process paired, three-arm measurements.",
        "primary_median_ms": {name: training["median_ms"] for name, training in trainings.items()
                              if name.startswith(("primary-", "best-"))},
        "robustness_speedups_vs_native": robustness,
        "temporary_peak_memory_fraction": memory,
        "memory_scope": "Temporary peak over two resident models; not single-model total memory.",
        "residual_case_summaries": {name: reports[name].get("case_summary") for name in reports if name.startswith("residual-")},
        "cross_entropy_case_summaries": {name: reports[name].get("case_summary") for name in reports if name.startswith("ce-")},
        "report_sha256": {name: sha(args.final_dir / (name + ".json")) for name in reports},
        "per_run_gates": {"identity": identity, "sources": sources, "correctness": correctness,
                          "measurements": measurements, "adapter_coverage": coverage},
        "dispatch_audit": {"path": str(args.dispatch_audit.resolve()),
                           "sha256": sha(args.dispatch_audit) if args.dispatch_audit.is_file() else None,
                           "passed": audited, "payload": audit},
        "issues": issues, "evidence_folder": str(args.final_dir.resolve()),
        "limitations": ["Synthetic pinned Megatron GPT steps; production training is not established.",
                        "Current-best controls run independently; normalized ratios are indicative controls, not direct paired comparisons.",
                        "Residual operator reports retain the frozen parent v3 contract and base verifier.",
                        "Autocast results must be interpreted using actual executed candidate dispatch; passing does not establish low-precision fusion."],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
