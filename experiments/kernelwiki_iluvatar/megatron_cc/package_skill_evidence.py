"""Add the completed stage-23 loop and its exact evidence to a skill checkout."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ID = "exp-bi-v150-corex42-stage23"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads((ROOT / "evidence/decision-0003.json").read_text())
    if decision["decision"] != "promote_for_tested_scope":
        raise ValueError("stage-23 packaging requires the recorded final gate result")
    skill = args.skill.resolve()
    bundle = skill / "evidence/bi-v150-corex42-stage23"
    bundle.mkdir(parents=True, exist_ok=False)
    selected = ["benchmark.py", "contract.json", "native_control.py", "run_remote.py", "summarize.py",
                "audit_autocast_dispatch.py", "probe_gpu_compare.py", "probe_native_precision.py",
                "references/compiled-launch-api.txt", "references/feedback_corrections.txt"]
    for candidate_id in ("0001", "0002", "0003"):
        selected += [f"candidates/{candidate_id}/candidate.py", f"candidates/{candidate_id}/candidate.py.receipt.json"]
    for name in ("decision-0002.json", "decision-0003.json", "gpu-comparator-probe.json",
                 "native-precision-probe.json", "0003-autocast-dispatch.json", "ixsmi.txt",
                 "runtime-source-sha256.txt", "tracefs-20261002.txt", "cc-final-review.json"):
        selected.append("evidence/" + name)
    for folder in ("0001-screen-v3", "0002-final", "0003-final"):
        selected += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "evidence" / folder).glob("*")) if p.is_file()]
    for relative in selected:
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, dest)
    # Keep the native loader's dependencies adjacent to benchmark.py so the
    # evidence bundle is runnable as well as independently hash-verifiable.
    for name in ("stage19_megatron_route_step.py", "stage13_backward.py", "stage3_residual_rmsnorm.py"):
        shutil.copyfile(ROOT / "references" / name, bundle / name)
    evidence = sorted(p for p in bundle.rglob("*") if p.is_file())
    receipt = {"source_id": SOURCE_ID, "recorded_at": "2026-10-02", "candidate_parent_chain": ["0001", "0002", "0003"],
               "files": [{"path": p.relative_to(bundle).as_posix(), "sha256": sha(p)} for p in evidence]}
    (bundle / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    evidence.append(bundle / "manifest.json")
    front = f'''---
id: {SOURCE_ID}
title: "BI-V150 native Megatron residual/RMSNorm Claude Code loop, 2026-10-02"
source_category: local-experiment
architectures: [bi-v150]
tags: [triton, kernel-fusion, reduction, jit-compilation]
recorded_at: 2026-10-02
environment:
  gpu: "Iluvatar BI-V150, device 0"
  corex: "4.2 stack; IX-ML and driver reported 4.2.0"
  torch: "2.4.1"
  triton: "2.1.0, CoreX vendor build"
  megatron: "5be9626709af2722333bf54797c954c09edeada3"
  controller: "Claude Code CLI 2.1.168 on Windows; Paratera requested model Claude-Opus-4.8"
evidence_files:
'''
    for path in evidence:
        front += f"  - path: {path.relative_to(skill).as_posix()}\n    sha256: {sha(path)}\n"
    body = '''---

# Scope and actual workflow

Claude Code CLI read the project-managed `kernelwiki-iluvatar` skill and produced
three original candidates. The controller ran on Windows; all GPU computation
ran through SSH on BI-V150 device 0. A fixed external harness supplied feedback.
This is a downstream KDA basic-flow implementation, not a run of the optional
Humanize plugin or a production training pipeline.

The baseline is the unmodified pinned Megatron local layer spec, native BDA and
`torch.nn.RMSNorm`. A process-local DTensor import mapping and incompatible-TE
mask allow that backend to import; vendor packages and Megatron are not edited.
TE is not a performance baseline. Attention BDA -> pre-MLP RMSNorm is replaced
in all four layers of a representative synthetic GPT: H512, 8 heads, sequence128,
micro-batch2, vocab1024, dropout0, no linear bias, epsilon1e-5, SGD. The holdout is
two layers, H256, 4 heads, sequence17, micro-batch2 and vocab512.

# Verifier repairs retained as evidence

Before candidate implementation, native FP16 RMSNorm backward at scale0.001
produced infinities. Those16 cases per round remain in the matrix as CPU checked
NaN/signed-infinity compatibility only, never finite-correctness passes.

An unequal-pair probe later found GPU float64 subtraction in this specific
CoreX/PyTorch build returned [0,0] for [1,2] minus [1.125,2.25]. GPU float32 and
CPU float64 returned [0.125,0.25]. This does not establish a hardware FP64 limit.
The verifier was repaired to compare CPU float64 copies, outside timed blocks,
without changing tolerances, inputs, baseline or measurement code. Prior GPU
float64 numerical claims are superseded; their raw records remain in the main
project. Both original candidates were rerun under contract v3.

# Candidate lineage and results

- 0001: standalone fused forward plus two-kernel backward, real epsilon and
  native fallback for low-precision residual. CPU checks passed; four-layer wall
  ratio0.899387, so it was rejected for training performance.
- 0002: cache compiled kernel code/launch metadata, use current-stream compiled
  launches and remove redundant views. Three FP32 training ratios1.005022,
  1.010870,1.021433, geomean1.012419. Autocast failed strict output/gradient
  checks, so the candidate was rejected. Host-dispatch attribution is a
  hypothesis; the diagnostic profile also included cold compilation.
- 0003: keep 0002's optimized path outside autocast, use exact native BDA and
  RMSNorm whenever CUDA autocast is active. All final gates passed for the tested
  configurations. The autocast dispatch audit observed zero compiled kernel
  cache entries after forward/backward and exact native values/gradients.

Final CPU numerical coverage:672/672 finite-native case-rounds plus48/48 separate
native-invalid compatibility case-rounds. Four shapes, five x/residual dtype
pairs, two epsilons, three input scales and both residual-gradient modes were
checked in three processes. Forward, loss, gradients, three SGD updates and
after-timing parameters also passed for the models.

The candidate's implementation guard allows rows<=2048 and hidden<=8192; this
is not a proof over every such shape. Tested operator shapes are16x256,256x512,
7x769,17x1537; tested model fusion shapes are128x2x512 and17x2x256.

# Timing and interpretation

Operators use warm GPU events,15 samples x30 calls, alternate order. Events can
include host launch gaps. Training uses synchronized `time.perf_counter` around
10 consecutive complete steps per sample,12 alternate-order samples and5 warmup
steps. The timed work includes zero_grad, forward, loss, backward and SGD,
excluding initial compilation, data loading and checkpoint I/O. Use baseline
median / candidate median; do not equate event latency with wall throughput.

The final decision JSON contains each process median and raw reports contain
every timing sample. Temporary peak allocation was equal for both paths within
the two-resident-model measurement scope; single-model total memory is untested.
The observed small improvement does not establish production, distributed,
TE, or low-precision fused-training gains. Autocast is native fallback only.

# Reproduction

With the pinned Megatron checkout and the same CoreX environment, enter the
`evidence/bi-v150-corex42-stage23` directory and run:

```bash
export PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages
export LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib
export CUDA_VISIBLE_DEVICES=0
python -u benchmark.py --megatron-root /path/to/pinned/megatron-lm \\
  --candidate candidates/0003/candidate.py --mode model --output new-model.json
```

For operator checks use `--mode operator`; output paths must not exist. The
original CLI prompts/replies, older verifier versions and full raw lineage are
also retained in the main project's `experiments/kernelwiki_iluvatar/megatron_cc`.
'''
    source = skill / "sources/experiments/bi-v150-corex42-stage23.md"
    source.write_text(front + body, encoding="utf-8")
    ratio = decision["primary_geomean"]
    page = f'''---
id: kernel-residual-rmsnorm-megatron-bi-v150
title: "Residual Add RMSNorm against native Megatron on BI-V150"
type: kernel
architectures: [bi-v150]
tags: [triton, kernel-fusion, reduction, jit-compilation]
kernel_types: [fused-kernel, reduction]
languages: [triton]
confidence: experimental
reproducibility: runnable
related: [kernel-residual-rmsnorm-bi-v150, kernel-residual-rmsnorm-backward-bi-v150, lang-triton-iluvatar, technique-kernel-fusion]
sources: [{SOURCE_ID}]
aliases: ["BI-V150 native Megatron fusion", "Claude Code BI-V150 optimization loop"]
performance_claims:
  - gpu: "Iluvatar BI-V150"
    dtype: fp32
    shape: "4-layer GPT; H512, heads8, seq128, micro-batch2, vocab1024; fusion128x2x512"
    metric: "3-process geomean complete-step wall throughput ratio vs pinned native Megatron local backend"
    value: {ratio:.12f}
    source_id: {SOURCE_ID}
    source_locator: "evidence/bi-v150-corex42-stage23/evidence/decision-0003.json#primary_geomean"
---

# Native backend contract first

Preserve native BDA's cast to the residual dtype before addition, its observable
residual sum, `norm.eps` and RMSNorm return dtype. FP32 weight does not imply a
low-precision output: this build returned FP32 normalized values for low-precision
residual. A fixed-epsilon prototype or an eager formula with different casts is
not a drop-in replacement for arbitrary native Megatron call sites.

Candidate0003 supports an FP32-residual/FP32-weight fused path outside autocast
and exact native fallback elsewhere. Three CPU-checked operator processes,
three four-layer complete-step processes and one two-layer holdout passed.
The observed FP32 wall ratio geomean was{ratio:.6f}, about{(ratio-1)*100:.2f}%
more tokens per second in this synthetic benchmark. Each round was faster,
but the gain is small; it does not demonstrate production training speedup.

# Autocast and native anomalies

Candidate0002 failed the strict autocast output and gradient gate despite passing
standalone mixed-dtype tests. Candidate0003 uses the native backend when autocast
is active. Its passing autocast measurement establishes compatibility and no
measured regression, not optimized BF16 training.

Native FP16 tiny-input gradients were nonfinite.16 cases per process therefore
check native-invalid compatibility, separately from224 finite-native cases.
In this build GPU float64 subtraction produced false zeros in a deliberate
unequal-pair probe; numerical comparisons now use CPU copies. These are observed
software-path limitations, not claims about absent GPU hardware features.

# Runnable entry point

Run this example from the skill root in the recorded BI-V150 toolchain:

```python
import importlib.util
from pathlib import Path
import torch

path = Path("evidence/bi-v150-corex42-stage23/candidates/0003/candidate.py")
spec = importlib.util.spec_from_file_location("biv150_candidate", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
x = torch.randn(256, 512, device="cuda", requires_grad=True)
residual = torch.randn_like(x, requires_grad=True)
weight = torch.ones(512, device="cuda", requires_grad=True)
y, summed = module.fused(x, residual, weight, 1e-5)
reference = torch.nn.functional.rms_norm(residual + x, (512,), weight, 1e-5)
torch.testing.assert_close(y.detach().cpu(), reference.detach().cpu(), atol=3e-4, rtol=1e-3)
(y.square().mean() + 0.01 * summed.square().mean()).backward()
```

The [source experiment](../../sources/experiments/bi-v150-corex42-stage23.md)
provides the exact native Megatron verifier, code hashes, all final raw samples,
failed parent candidates and measurement scope. Cache only compiled code and
metadata, never computed tensor results; retain the current stream and all
required compilation specializations. The installed CoreX compiled-launch API
is version-specific and does not establish an NVIDIA hardware equivalence.
'''
    (skill / "wiki/kernels/residual-rmsnorm-megatron-bi-v150.md").write_text(page, encoding="utf-8")
    for relative in ("wiki/hardware/bi-v150-stack.md", "wiki/languages/triton-iluvatar.md",
                     "wiki/kernels/residual-rmsnorm-bi-v150.md", "wiki/kernels/residual-rmsnorm-backward-bi-v150.md"):
        path = skill / relative
        content = path.read_text(encoding="utf-8")
        start = content.index("sources: [")
        end = content.index("]", start)
        content = content[:end] + ", " + SOURCE_ID + content[end:]
        content += "\n## Native Megatron loop, 2026-10-02\n\n" + (
            "[Stage23](../../sources/experiments/bi-v150-corex42-stage23.md) adds CPU-copy\n"
            "numerical verification, real epsilon/dtype semantics, a three-candidate\n"
            "Claude Code loop and four-layer complete-step wall measurements. Read the\n"
            "[native-backend kernel page](../kernels/residual-rmsnorm-megatron-bi-v150.md)\n"
            "for tested scope, autocast native fallback and observed software anomalies.\n")
        path.write_text(content, encoding="utf-8")
    ledger = skill / "references/iluvatar-migration-ledger.md"
    content = ledger.read_text(encoding="utf-8").replace("PROJECT_REPORT_AND_NEXT_PLAN.md", "PROJECT_REPORT_0930.md")
    content += f'''\n## 原生 Megatron 与 Claude Code 闭环（2026-10-02）

`exp-bi-v150-corex42-stage23` 增加真实原生后端对照：Claude Code CLI 使用项目级
skill 生成三个候选，BI-V150 上由固定程序执行。最终四层 FP32 合成 GPT 三轮
完整 step 墙钟倍率为1.011554/1.016174/1.004844，几何平均{ratio:.6f}；
仅能说明所测小模型约{(ratio-1)*100:.2f}% 的吞吐改善。三轮672个有限值算子
case-round通过，48个原生异常case-round单独核对兼容性。autocast 使用原生
回退，不是融合 BF16 训练的正面证据。

新证据还记录了实际 epsilon、BDA 类型转换与 RMSNorm 输出类型，以及当前
CoreX GPU float64 比较返回零误差的异常。验收已改为 CPU 比较且容差不变。
阶段21/22的 GPU Event 计时未包含 zero_grad；阶段23完整 step 使用同步墙钟
并包含清梯度、前向、loss、反向和SGD。不同定义的数据不能合并为同一训练指标。
下一步仍是生产配置的形状、精度与稳定吞吐；不把小模型结果外推为生产训练收益。
'''
    ledger.write_text(content, encoding="utf-8")
    index = skill / "index.md"
    content = index.read_text(encoding="utf-8").replace("[one-layer GPT fusion evidence]", "[legacy one-layer GPT fusion evidence]")
    content = content.replace("[GEMM bias epilogue]", "[native Megatron CLI loop](wiki/kernels/residual-rmsnorm-megatron-bi-v150.md), [GEMM bias epilogue]")
    index.write_text(content, encoding="utf-8")
    print(json.dumps({"source_id": SOURCE_ID, "evidence_files": len(evidence), "primary_geomean": ratio}))


if __name__ == "__main__":
    main()
