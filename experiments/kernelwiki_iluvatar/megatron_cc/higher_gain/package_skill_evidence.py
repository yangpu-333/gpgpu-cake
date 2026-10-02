"""Package a promoted stage-24 CE extension and its exact evidence into a skill.

This script does not tune kernels, change experiment records or regenerate
query indexes. Run scripts/generate-indices.py and the validator after packaging.
"""

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
SOURCE_ID = "exp-bi-v150-corex42-stage24"
KERNEL_ID = "kernel-cross-entropy-megatron-bi-v150"
BUNDLE_NAME = "bi-v150-corex42-stage24"
SEEDS = (24002, 24003, 24004)
RUN_NAMES = ([f"{arm}-{seed}" for arm in ("residual", "ce", "primary", "best") for seed in SEEDS]
             + ["holdout-24005", "autocast-24006"])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def positive_number(value, name):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError("invalid recorded " + name)
    return value


def count_summaries(decision, field):
    summaries = decision[field].values()
    return {name: sum(item[name] for item in summaries) for name in
            ("finite_native_cases", "finite_correctness_passed",
             "native_invalid_cases", "native_invalid_compatibility_passed")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--dispatch-audit", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9]{4}", args.candidate_id):
        parser.error("four-digit candidate id required")
    skill = args.skill.resolve()
    bundle = skill / "evidence" / BUNDLE_NAME
    source_path = skill / "sources/experiments/bi-v150-corex42-stage24.md"
    kernel_path = skill / "wiki/kernels/cross-entropy-megatron-bi-v150.md"
    if bundle.exists() or source_path.exists() or kernel_path.exists():
        raise FileExistsError("stage-24 destination exists; packaging is not an overwrite operation")
    decision = read_json(args.decision)
    if decision.get("decision") != "promote_for_tested_scope":
        raise ValueError("stage-24 packaging requires recorded promotion")
    if decision.get("candidate_id") != args.candidate_id:
        raise ValueError("candidate id does not match decision")
    if not decision.get("checks") or not all(value is True for value in decision["checks"].values()):
        raise ValueError("promotion checks are not all true")
    native_ratio = positive_number(decision["primary_geomean_vs_native"], "native geomean")
    normalized_ratio = positive_number(decision["normalized_geomean_vs_current_best"], "normalized control geomean")
    contract = read_json(ROOT / "contract.json")
    candidate_path = ROOT / "candidates" / args.candidate_id / "candidate.py"
    candidate_receipt_path = candidate_path.with_suffix(".py.receipt.json")
    receipt = read_json(candidate_receipt_path)
    for field, path in (("candidate_sha256", candidate_path),
                        ("best_candidate_sha256", PARENT / "candidates/0003/candidate.py"),
                        ("base_harness_sha256", PARENT / "benchmark.py"),
                        ("parent_contract_sha256", PARENT / "contract.json"),
                        ("extension_harness_sha256", ROOT / "driver.py"),
                        ("contract_sha256", ROOT / "contract.json")):
        if decision.get(field) != sha(path):
            raise ValueError("source hash differs from promoted decision: " + field)
    if receipt.get("output_sha256") != sha(candidate_path):
        raise ValueError("candidate extraction receipt does not match source")
    audit = read_json(args.dispatch_audit)
    if (audit.get("passed") is not True or audit.get("candidate_sha256") != sha(candidate_path)
            or decision.get("dispatch_audit", {}).get("sha256") != sha(args.dispatch_audit)):
        raise ValueError("dispatch audit does not match promoted decision")
    final_dir = args.final_dir.resolve()
    reports = {}
    for name in RUN_NAMES:
        path = final_dir / (name + ".json")
        reports[name] = read_json(path)
        if reports[name].get("status") != "passed" or decision["report_sha256"].get(name) != sha(path):
            raise ValueError("final raw report does not match decision: " + name)
    date = contract["date"]
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date):
        raise ValueError("contract date is not an ISO date")
    residual_summary = count_summaries(decision, "residual_case_summaries")
    ce_summary = count_summaries(decision, "cross_entropy_case_summaries")
    autocast = reports["autocast-24006"]["model"]
    ce_observations = autocast["candidate_calls"]["cross_entropy"]["observations"]
    residual_observations = autocast["candidate_calls"]["observations"]
    if not ce_observations or not residual_observations:
        raise ValueError("autocast dtype observations missing")
    observed_ce = ce_observations[0]
    observed_residual = residual_observations[0]
    autocast_checks = audit.get("autocast_exact_native_checks", [])
    autocast_exact = (bool(autocast_checks) and isinstance(autocast_checks, list)
                      and all(isinstance(item, dict) and all(
                          isinstance(item.get(key), dict) and item[key].get("passed") is True
                          and item[key].get("atol") == 0 and item[key].get("rtol") == 0
                          for key in ("loss_exact_native", "dlogits_exact_native"))
                              for item in autocast_checks))
    if autocast_exact and audit.get("autocast_triton_runner_launches") == 0:
        audited_dtypes = ", ".join(sorted({item["input_dtype"] for item in autocast_checks}))
        autocast_description = (
            "For the promoted candidate, CE uses native-equivalent TP=1 Torch autograd\n"
            "operations inside active CUDA autocast, omitting the singleton all-reduce\n"
            "identities. The independent audit recorded exact native loss and dlogits\n"
            f"for its {audited_dtypes} input cases and zero CE Triton runner launches\n"
            "in that context. Both the inherited residual path and this CE autocast\n"
            "branch therefore differ from the optimized outside-autocast Triton path.\n"
            "The adapter's `optimized` CE count records candidate API selection, not\n"
            "Triton execution. This is compatibility through native-equivalent math;\n"
            "it does not establish optimized BF16 CE training or production throughput.\n")
        kernel_autocast_description = (
            "Under active CUDA autocast, the promoted CE candidate uses\n"
            "native-equivalent TP=1 Torch autograd operations with singleton\n"
            "all-reduce identities omitted. Independent audit cases had exact native\n"
            "loss/dlogits and zero CE Triton runner launches. The inherited\n"
            "residual/RMSNorm also uses native fallback under autocast. Adapter\n"
            "API counters do not prove Triton execution or optimized BF16 training.\n")
        ledger_autocast_description = (
            "autocast下残差/RMSNorm仍使用原生回退；CE候选使用原生等价的TP=1\n"
            "Torch autograd计算并省略单成员all-reduce恒等调用。独立审计中的loss和\n"
            "dlogits与原生完全一致，CE Triton runner调用数为0。adapter计数只表示\n"
            "候选API选择，不表示融合BF16 CE训练已经优化。\n")
    else:
        autocast_description = (
            "CE remains eligible through the supplemental adapter. Its `optimized`\n"
            "count records candidate API selection, not a particular implementation\n"
            "branch or Triton execution. Read the actual dtype and dispatch evidence\n"
            "separately from residual fallback. Passing autocast is a compatibility\n"
            "and no-regression gate, not optimized BF16 CE training evidence.\n")
        kernel_autocast_description = (
            "The inherited residual/RMSNorm uses native fallback under autocast.\n"
            "CE adapter API counts must be interpreted with actual dispatch audits;\n"
            "they do not establish optimized BF16 CE training.\n")
        ledger_autocast_description = (
            "autocast下残差/RMSNorm仍使用原生回退；CE须按实际dtype和独立dispatch\n"
            "证据解释，adapter API计数不能证明Triton执行或BF16训练优化。\n")

    # Build the full plan before creating the destination. All experiment bytes
    # are copied verbatim; path remapping makes the recorded verifier runnable.
    selected = {}

    def add(path, destination, required=True):
        path = Path(path).resolve()
        if not path.is_file():
            if required:
                raise FileNotFoundError(path)
            return
        relative = Path(destination)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe evidence destination")
        key = relative.as_posix()
        if key in selected and selected[key] != path:
            raise ValueError("evidence destination collision: " + key)
        selected[key] = path

    for name in ("benchmark.py", "contract.json", "native_control.py", "invoke_cc.ps1", "extract_cli.py"):
        add(PARENT / name, name)
    for name in ("stage19_megatron_route_step.py", "stage13_backward.py", "stage3_residual_rmsnorm.py"):
        add(PARENT / "references" / name, name)
    for name in ("candidate.py", "candidate.py.receipt.json"):
        add(PARENT / "candidates/0003" / name, "candidates/0003/" + name)
    for name in ("contract.json", "driver.py", "run_extended.py", "summarize_extended.py",
                 "audit_ce_dispatch.py", "profile_native.py", "package_skill_evidence.py",
                 "audit_delivery.py"):
        add(ROOT / name, "higher_gain/" + name)
    response_name = str(receipt["response"]).replace("\\", "/").rsplit("/", 1)[-1]
    response_path = ROOT / "evidence" / response_name
    if receipt.get("response_sha256") != sha(response_path):
        raise ValueError("original candidate CLI response does not match extraction receipt")
    add(response_path, "higher_gain/evidence/" + response_name)
    for relative in ("evidence/cc-plan.json", "docs/draft.md", "docs/draft.md.receipt.json"):
        add(ROOT / relative, "higher_gain/" + relative)
    add(args.decision, "higher_gain/evidence/" + args.decision.name)
    add(args.dispatch_audit, "higher_gain/evidence/" + args.dispatch_audit.name)
    for name in RUN_NAMES:
        for extension in (".json", ".stdout", ".stderr"):
            add(final_dir / (name + extension), "higher_gain/evidence/" + final_dir.name + "/" + name + extension)
    add(final_dir / "commands.json", "higher_gain/evidence/" + final_dir.name + "/commands.json")
    for directory in ("prompts", "docs", "references"):
        for path in sorted((ROOT / directory).rglob("*")):
            if path.is_file() and path.suffix in (".txt", ".md", ".json", ".py"):
                add(path, "higher_gain/" + path.relative_to(ROOT).as_posix())
    # Include candidate parents and failed smoke/screen/final records if they
    # exist, while excluding profiler traces, binary .prof and cache artifacts.
    for path in sorted((ROOT / "candidates").glob("[0-9][0-9][0-9][0-9]/*")):
        if (path.is_file() and int(path.parent.name) <= int(args.candidate_id)
                and path.name in ("candidate.py", "candidate.py.receipt.json")):
            add(path, "higher_gain/" + path.relative_to(ROOT).as_posix())
    for path in sorted((ROOT / "evidence").rglob("*")):
        audit_source = re.fullmatch(r"([0-9]{4})-audit-source\.py", path.name)
        if path.is_file() and (path.name == "audit-ce-dispatch-preflight.py"
                               or (audit_source and int(audit_source.group(1)) <= int(args.candidate_id))):
            add(path, "higher_gain/" + path.relative_to(ROOT).as_posix())
            continue
        if not path.is_file() or path.suffix not in (".json", ".stdout", ".stderr"):
            continue
        relative = path.relative_to(ROOT / "evidence")
        experiment_folder = re.fullmatch(r"([0-9]{4})-(smoke|screen|final)", relative.parts[0])
        numbered_evidence = re.match(r"([0-9]{4})-", path.name)
        if ((len(relative.parts) == 1 and (
                path.name.startswith("cc-") or path.name.startswith("decision-")
                or (numbered_evidence and int(numbered_evidence.group(1)) <= int(args.candidate_id))))
                or (experiment_folder and int(experiment_folder.group(1)) <= int(args.candidate_id))):
            add(path, "higher_gain/" + path.relative_to(ROOT).as_posix())
    for name in ("higher-gain-native-profile.json", "higher-gain-native-cpu-profile.json",
                 "0001-host-profile.json", "decision-0003.json", "0003-autocast-dispatch.json",
                 "gpu-comparator-probe.json", "native-precision-probe.json", "ixsmi.txt",
                 "cc-candidate-0003.json"):
        add(PARENT / "evidence" / name, "evidence/" + name, required=False)
    # Require the specifically promoted candidate and extraction receipt even
    # if an unusual directory listing would not select them above.
    add(candidate_path, "higher_gain/candidates/" + args.candidate_id + "/candidate.py")
    add(candidate_receipt_path, "higher_gain/candidates/" + args.candidate_id + "/candidate.py.receipt.json")

    previous_kernel = skill / "wiki/kernels/residual-rmsnorm-megatron-bi-v150.md"
    ledger = skill / "references/iluvatar-migration-ledger.md"
    previous_text = previous_kernel.read_text(encoding="utf-8")
    ledger_text = ledger.read_text(encoding="utf-8")
    if not re.search(r"^sources: \[[^\n]*\]", previous_text, re.MULTILINE):
        raise ValueError("existing Megatron kernel source references are not an inline list")
    bundle.mkdir(parents=True, exist_ok=False)
    for relative, path in sorted(selected.items()):
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    copied = sorted(path for path in bundle.rglob("*") if path.is_file())
    manifest = {"source_id": SOURCE_ID, "recorded_at": date, "candidate_id": args.candidate_id,
                "decision": decision["decision"], "candidate_parent": receipt.get("parent"),
                "files": [{"path": path.relative_to(bundle).as_posix(), "sha256": sha(path)} for path in copied],
                "scope": "Exact experiment bytes; profiler JSON is diagnostic with overhead, not promotion timing."}
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    copied.append(manifest_path)
    decision_relative = "evidence/" + BUNDLE_NAME + "/higher_gain/evidence/" + args.decision.name
    environment = reports["primary-24002"]["environment"]
    controller_models = ", ".join(receipt.get("cli_model", [])) or "see original CLI extraction receipt"
    front = f'''---
id: {SOURCE_ID}
title: "BI-V150 native Megatron cross-entropy extension, {date}"
source_category: local-experiment
architectures: [bi-v150]
tags: [triton, kernel-fusion, reduction, jit-compilation]
recorded_at: {date}
environment:
  gpu: "{environment['gpu']}, device 0"
  corex: "{contract['expected_stack']['corex']} stack; consult retained environment evidence"
  torch: "{environment['torch']}"
  triton: "{environment['triton']}, CoreX vendor build"
  megatron: "{environment['megatron_commit']}"
  controller: "Claude Code CLI on Windows; Paratera requested model {controller_models}"
evidence_files:
'''
    for path in sorted(copied):
        front += f"  - path: {path.relative_to(skill).as_posix()}\n    sha256: {sha(path)}\n"
    table = "| Seed | Native ms | Extension ms | Extension/native | Normalized vs 0003 |\n|---|---:|---:|---:|---:|\n"
    for index, seed in enumerate(SEEDS):
        medians = decision["primary_median_ms"][f"primary-{seed}"]
        table += (f"| {seed} | {medians['native']:.6f} | {medians['candidate']:.6f} | "
                  f"{decision['primary_speedups_vs_native'][index]:.6f} | "
                  f"{decision['normalized_speedups_vs_current_best'][index]:.6f} |\n")
    body = f'''---

# Scope and baseline

Claude Code CLI used the project-managed KernelWiki adaptation to produce
candidate {args.candidate_id}. The controller ran on Windows and all GPU work
ran on BI-V150 device 0. The parent residual/RMSNorm implementation is retained;
the extension replaces only the optimized model's `compute_language_model_loss`.
The native model and installed Megatron/CoreX packages remain the baseline.
This is a downstream KDA basic-flow implementation, not the optional Humanize
plugin or an end-to-end production data/checkpoint pipeline.

The same pinned local PyTorch backend and synthetic four-layer FP32 GPT are
used: H512, heads8, sequence128, micro-batch2, vocab1024, RMSNorm epsilon1e-5,
dropout0, no linear bias and SGD. The holdout remains two layers, H256, heads4,
sequence17, micro-batch2, vocab512. Transformer Engine is not the baseline.

# Native cross-entropy semantics

The optimized CE path is restricted to TP=1, zero label smoothing, supported
floating CUDA logits `[S,B,V]` and int64 labels `[S,B]` with contiguous layout.
Unsupported calls use the captured native model method; TP>1 is not optimized.
Tested vocab sizes are 1, 7, 512, 769 and 1024, not every possible vocabulary.
Math runs in FP32 after the native input conversion and returns FP32 per-token
loss. Compute `log(sum(exp(logits-max))) - (target_logit-max)` for valid labels.
For native out-of-range labels, subtract zero and preserve softmax times dloss
in backward; `-100` is not a PyTorch ignore-index in this native implementation.
FP32 forward inputs are overwritten with saved probabilities as in native CE;
low-precision original logits remain unchanged. Backward may overwrite saved
probabilities; repeated-backward compatibility is not promised.

Standalone checks cover five shapes, three dtypes, seven finite distributions,
two label patterns and three layouts in each of three independent processes.
CE finite correctness is {ce_summary['finite_correctness_passed']}/{ce_summary['finite_native_cases']}
case-rounds; separate native-invalid layout compatibility is
{ce_summary['native_invalid_compatibility_passed']}/{ce_summary['native_invalid_cases']}.
The parent residual matrix remains
{residual_summary['finite_correctness_passed']}/{residual_summary['finite_native_cases']} finite
case-rounds and {residual_summary['native_invalid_compatibility_passed']}/{residual_summary['native_invalid_cases']}
native-nonfinite compatibility case-rounds. All numerical comparisons use CPU
float64 copies with the unchanged tolerances. Output, loss, all parameter
gradients, three SGD updates and after-timing parameters pass in model checks.

# Complete-step measurements

Training timing is the unchanged parent synchronized `time.perf_counter`
protocol: five warmup steps, twelve alternating-order samples of ten complete
steps, including zero_grad, forward, loss, backward and SGD. Initial compilation,
data loading and checkpoint I/O are excluded. Standalone CE has no timing
benchmark because native forward/backward mutate inputs.

{table}

The three-process native-relative geomean is {native_ratio:.12f}, corresponding
to {(native_ratio - 1) * 100:.2f}% more tokens per second in this tested benchmark.
The normalized geomean against current best 0003 is {normalized_ratio:.12f}.
Each extension and 0003 control is paired with its own native measurement in
separate processes. `(extension/native)/(0003/native)` is an independent
normalized comparison, not a directly paired three-arm measurement.
See `{decision_relative}` for the exact gate result and all process medians.
Temporary peak memory is measured over two resident models; it is not total
single-model memory.

# Autocast, dispatch and profile limits

The autocast report observed CE logits `{observed_ce['logits']['dtype']}` with
shape `{observed_ce['logits']['shape']}` and stride `{observed_ce['logits']['stride']}`;
loss was `{observed_ce['loss']['dtype']}` with shape `{observed_ce['loss']['shape']}`.
Residual fusion inputs were x `{observed_residual['x_dtype']}`, residual
`{observed_residual['residual_dtype']}` and normalized output `{observed_residual['y_dtype']}`.
The inherited 0003 residual/RMSNorm implementation uses exact native fallback
inside active CUDA autocast.

{autocast_description}

The independent dispatch audit observed compiled CE forward and backward runner
invocations on FP32 `[128,2,1024]` outside autocast and promotion timed regions. Cache format
keys and runner names establish this software execution path, not specific
NVIDIA instructions or hardware equivalence. Diagnostic profiler JSON retains
its explicit overhead scope; profiling and nested CPU/device sums are not used
as promotion throughput or exclusive GPU attribution.

# Reproduction

From `evidence/{BUNDLE_NAME}` with the recorded toolchain and exact pinned
Megatron checkout, use fresh output paths:

```bash
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib
export CUDA_VISIBLE_DEVICES=0
python -u higher_gain/driver.py --megatron-root /path/to/pinned/megatron-lm \\
  --candidate higher_gain/candidates/{args.candidate_id}/candidate.py \\
  --variant extended --mode cross-entropy --output new-ce.json
python -u higher_gain/driver.py --megatron-root /path/to/pinned/megatron-lm \\
  --candidate higher_gain/candidates/{args.candidate_id}/candidate.py \\
  --variant extended --mode model --output new-model.json
```

For the contemporaneous 0003 control use `--variant current-best` and
`--candidate candidates/0003/candidate.py`. Use the parent `benchmark.py`
operator mode for the unchanged residual matrix. Final raw records, original
CLI plan/response/receipt, failed parents if present, scripts and byte hashes
are preserved in this bundle. Raw CLI plan statements remain unchanged; the
execution correction JSON and promoted dispatch audit define the actual final
autocast behavior. No production, TP>1, TE or multi-GPU gain is proved.
'''
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(front + body, encoding="utf-8")
    page = f'''---
id: {KERNEL_ID}
title: "Native Megatron vocabulary cross-entropy fusion on BI-V150"
type: kernel
architectures: [bi-v150]
tags: [triton, kernel-fusion, reduction, jit-compilation]
kernel_types: [fused-kernel, reduction]
languages: [triton]
confidence: experimental
reproducibility: runnable
related: [kernel-residual-rmsnorm-megatron-bi-v150, lang-triton-iluvatar, technique-kernel-fusion]
sources: [{SOURCE_ID}]
aliases: ["BI-V150 Megatron cross entropy", "singleton TP vocabulary loss fusion"]
performance_claims:
  - gpu: "Iluvatar BI-V150"
    dtype: fp32
    shape: "4-layer GPT; H512, heads8, seq128, micro-batch2, vocab1024; CE128x2x1024"
    metric: "3-process geomean complete-step wall throughput ratio vs pinned native Megatron local backend; residual/RMSNorm plus CE"
    value: {native_ratio!r}
    source_id: {SOURCE_ID}
    source_locator: "{decision_relative}#primary_geomean_vs_native"
---

# Optimize the actual native loss route

For TP=1, the native vocabulary loss still executes three singleton all-reduce
calls and multiple pointwise/reduction operations. This extension replaces only
the optimized model's CE call, retaining candidate0003 residual/RMSNorm and the
same native baseline, model, batches, optimizer, checks and full-step timer.
The promoted synthetic FP32 complete-step ratio is {native_ratio:.6f}; it is a
combined pipeline gain, not an isolated CE kernel speedup.

Preserve max-shift FP32 loss math, observable FP32 input overwrite into
probabilities, unchanged low-precision inputs and native out-of-range-label
gradients. Native `-100` does not mean ignore-index. The implemented path is
TP=1 and zero smoothing only; unsupported layouts and TP groups use native
fallback. Tested vocab sizes 1/7/512/769/1024 do not certify an unrestricted range.

# Runnable entry point

From the skill root in the recorded CoreX environment:

```python
import importlib.util
from pathlib import Path
import torch

path = Path("evidence/{BUNDLE_NAME}/higher_gain/candidates/{args.candidate_id}/candidate.py")
spec = importlib.util.spec_from_file_location("biv150_ce_candidate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
logits = torch.randn(128, 2, 1024, device="cuda", requires_grad=True)
target = torch.randint(1024, (128, 2), device="cuda")
loss = module.cross_entropy(logits, target)
assert loss.shape == target.shape and loss.dtype == torch.float32
loss.mean().backward()
```

Use the [source experiment](../../sources/experiments/bi-v150-corex42-stage24.md)
for native output/gradient/mutation verification, all raw timing samples and
exact source hashes. The BF16-autocast report has its actual CE dtype observations;
{kernel_autocast_description}

Independent 0003 controls are normalized to their own native runs, not directly paired to
the extension. CPU-copy comparisons avoid the archived CoreX GPU float64
comparator anomaly without changing tolerances.

The compiled-runner audit verifies CE execution for the main FP32 shape only.
No tcgen05/WGMMA/TMA equivalence, production speedup, TP>1 optimization or
distributed scaling follows from this evidence.
'''
    kernel_path.parent.mkdir(parents=True, exist_ok=True)
    kernel_path.write_text(page, encoding="utf-8")
    previous_text = re.sub(r"^sources: \[([^\n]*)\]",
                           lambda match: "sources: [" + match.group(1) + ", " + SOURCE_ID + "]",
                           previous_text, count=1, flags=re.MULTILINE)
    previous_text += (f"\n## Vocabulary loss extension, {date}\n\n"
                      "[Stage24](../../sources/experiments/bi-v150-corex42-stage24.md) preserves this\n"
                      "residual implementation and adds TP=1 native vocabulary cross-entropy fusion.\n"
                      "The new complete-step result combines both operators; it does not replace\n"
                      "this page's historical stage23 measurement or extend its residual dtype scope.\n"
                      "See [the CE kernel page](cross-entropy-megatron-bi-v150.md) for tested semantics.\n")
    previous_kernel.write_text(previous_text, encoding="utf-8")
    ledger_text += f'''\n## 原生 Megatron 词表交叉熵融合（阶段24，{date}）

`{SOURCE_ID}` 保留0003残差/RMSNorm代码，扩充TP=1、零label smoothing的
词表交叉熵候选{args.candidate_id}。四层FP32合成GPT配置、原生后端、输入、容差、
SGD和完整step墙钟计时均保持一致。原生对照三轮几何平均倍率为{native_ratio:.6f}，
即所测任务约{(native_ratio - 1) * 100:.2f}%吞吐改善；相对0003的独立归一化
对照倍率为{normalized_ratio:.6f}，不是同一进程的三臂配对测量。

CE有限值case-round共{ce_summary['finite_correctness_passed']}/{ce_summary['finite_native_cases']}通过，
原生布局错误兼容性另计{ce_summary['native_invalid_compatibility_passed']}/{ce_summary['native_invalid_cases']}。
原生FP32输入修改、低精度输入不变、越界标签行为和反向梯度均纳入CPU副本验证。
{ledger_autocast_description}
独立审计记录主FP32形状的CE前向/反向compiled runner，不能据此声称NVIDIA指令等价。
原始证据保留在`evidence/{BUNDLE_NAME}`；诊断profile有额外开销，不能替代验收计时。
这些结果仍限于固定小模型，生产、TP>1、多卡和TE收益尚未验证。
'''
    ledger.write_text(ledger_text, encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "kernel_id": KERNEL_ID,
                      "evidence_files": len(copied), "candidate_id": args.candidate_id,
                      "primary_geomean_vs_native": native_ratio,
                      "normalized_geomean_vs_current_best": normalized_ratio,
                      "next": "regenerate query indexes and validate the skill"}, indent=2))


if __name__ == "__main__":
    main()
