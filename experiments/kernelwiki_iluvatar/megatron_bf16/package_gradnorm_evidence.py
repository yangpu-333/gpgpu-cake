"""Add the measured native gradient-norm staging result to the project skill."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE='exp-bi-v150-corex42-stage26'
BUNDLE='bi-v150-corex42-stage26'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',type=Path,required=True);a=p.parse_args()
    decision=json.loads((ROOT/'evidence/gradnorm-stage26/decision.json').read_text())
    assert decision['status']=='accepted_for_tested_main_scope'
    bundle=a.skill/'evidence'/BUNDLE;bundle.mkdir(exist_ok=False)
    (bundle/'.gitattributes').write_text('* -text -eol -whitespace\n')
    selected=[]
    for name in ('probe_gradnorm.py','launch_gradnorm.py','compare_gradnorm.py','run_gradnorm_performance.py','gradnorm-shapes.json','backup_gradnorm.py','verify_gradnorm_receipts.py','launch_training.py','kernel_adapter.py','run_case.py','compare_training.py','contracts/0008-main.json','candidates/0008/candidate.py'):
        path=ROOT/name
        if path.exists():selected.append(path)
    selected+=list((ROOT/'gradnorm_candidates').rglob('*'))
    selected+=[ROOT/'prompts/cc-gradnorm-001.txt',ROOT/'evidence/cc-gradnorm-001.json',ROOT/'evidence/gradnorm-probe-001.json']
    selected+=list((ROOT/'evidence/gradnorm-stage26').rglob('*'))
    records=[]
    for path in sorted(selected):
        rel=path.relative_to(ROOT)
        if not path.is_file() or 'snapshots' in rel.parts or '__pycache__' in rel.parts or path.suffix=='.gz':continue
        dest=bundle/'megatron_bf16'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        records.append({'path':'megatron_bf16/'+rel.as_posix(),'sha256':sha(dest),'bytes':dest.stat().st_size})
    for name in ('runtime_compat.py','source_manifest.json','summarize_performance.py'):
        path=ROOT.parent/'megatron_training_entry'/name
        dest=bundle/'megatron_training_entry'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        records.append({'path':'megatron_training_entry/'+name,'sha256':sha(dest),'bytes':dest.stat().st_size})
    (bundle/'manifest.json').write_text(json.dumps({'source_id':SOURCE,'files':records},indent=2)+'\n')
    front=f'''---
id: {SOURCE}
title: "BI-V150 formal BF16 Megatron gradient L2 norm host-transfer batching"
source_category: local-experiment
architectures: [bi-v150]
tags: [reduction]
recorded_at: 2026-10-03
environment:
  gpu: "Iluvatar BI-V150, device 0"
  corex: "4.2.0"
  torch: "2.4.1"
  megatron: "5be9626709af2722333bf54797c954c09edeada3"
  controller: "Claude Code CLI / Paratera Claude-Opus-4.8"
evidence_files:
'''
    for path in sorted(bundle.rglob('*')):
        if path.is_file():front+=f'  - path: {path.relative_to(a.skill).as_posix()}\n    sha256: {sha(path)}\n'
    body=f'''---

# Optimize the measured native backend

The pinned local `local_multi_tensor_l2_norm` computes every per-parameter GPU
`torch.norm`, constructs `torch.tensor(l2)` on CPU (implicitly reading individual
GPU scalar norms), reduces the CPU vector, and returns a one-element FP32 CUDA
tensor. Candidate001 preserves the same GPU norm calls/order and final CPU
reduction; it stacks scalar norms and makes one batched CPU copy. This is host
synchronization/transfer batching around a GPU reduction, not a new Triton
compute kernel, faster GPU reduction or BI-V150 hardware instruction equivalence.
It addresses this recorded local fallback, not an optimized TE/Apex backend.

Claude Code generated the candidate using the project skill. Only its import
and two candidate functions were extracted verbatim; its self-tests were
excluded and a separate fixed verifier was used. Candidate SHA256 is
`{decision['candidate_sha256']}`. Unsupported lists/dtypes/layouts/devices or
per-tensor requests delegate to the supplied original native function.
Finite FP32 CUDA:0 contiguous ordinary gradients are the tested path.

# Correctness and actual training

Three recorded gradient lists (28/28/52 tensors) and four scales
0/0.001/1/100 passed all12 standalone scalar-bit-equality and input-preservation
cases. Only main (4layers/H512/S128/V1024/batch2) has formal training validation:
all1475 three-update loss/gradient/BF16-model/FP32-SGD-master checks passed,
and every compared tensor was bitwise equal to the CE0008 control. Actual
training gradient norms were additionally compared to the native norm function
on the same gradients every step, with all three FP32 scalar bit patterns equal.
CPUfloat64 comparisons retain atol3e-4/rtol1e-3. These are three updates,
not production data or long-run convergence. Separate120step coverage records
zero native fallback; API counts do not prove individual GPU kernel launches.

# Formal throughput

Nine successful120step processes use three seeds and sequential fresh native,
CE0008-control, CE0008-plus-norm arms, alternating arm order. Original formal
GPU-synchronized `time.time` intervals retain ten interval10 means ending30..120;
first20steps, profiling, snapshots, evaluation and checkpoint are excluded.
Same-trial geomean throughput ratios are **1.03872346 vs CE0008** (+3.87%)
and **1.04409785 vs original native** (+4.41%). Every seed is faster in both
comparisons. Three seeds are observations, not confidence intervals. Do not
compare or multiply the historical stage25 4.62% from a separate batch.

| Seed | Native ms/step | CE0008 control | CE0008 + norm batching |
|---|---:|---:|---:|
|26102|31.70|30.44|29.67|
|26103|30.44|32.17|29.73|
|26104|30.59|29.68|29.40|

The norm-only microbenchmark is a synchronized wall interval that includes
host synchronization. It is not a CUDA Event-only device metric or formal
training throughput. Original failed launcher/metadata gates are retained.
No native source, CE0008 candidate, tolerance, timer or optimizer math changed.

# Runnable candidate

```python
import importlib.util
from pathlib import Path
import torch
path = Path("evidence/{BUNDLE}/megatron_bf16/gradnorm_candidates/001/candidate.py")
spec = importlib.util.spec_from_file_location("gradnorm", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
grads = [torch.randn(512, 512, device="cuda", dtype=torch.float32)]
norm, auxiliary = module.local_multi_tensor_l2_norm(2048, None, [grads], False)
assert auxiliary is None and norm.shape == (1,)
```

Full code, contracts via stage25, outer wrappers, actual norm observations,
raw intervals and hash receipts are in `evidence/{BUNDLE}`. Raw tensor files
are separately backed up; snapshot receipts are included, large tensors are not.
This result does not establish wide/deep formal gains, AdamW, clipping,
nonfinite-input compatibility, TP>1, multi-GPU, production or NVIDIA equivalence.
'''
    page=a.skill/'sources/experiments/bi-v150-corex42-stage26.md';assert not page.exists();page.write_text(front+body)
    skill=a.skill/'SKILL.md';text=skill.read_text();text=text.replace('## Knowledge Base Contents','8. **For native Megatron gradient L2 norms, inspect host scalar transfers.** [Stage26](sources/experiments/bi-v150-corex42-stage26.md) batches host copies while retaining native GPU norm operations. Its main-model BF16 training gains use same-trial CE0008 controls; do not label them new Triton reduction speedups or extend them to TE/Apex and untested training settings.\n\n## Knowledge Base Contents');skill.write_text(text)
    ledger=a.skill/'references/iluvatar-migration-ledger.md';ledger.write_text(ledger.read_text()+f'\n## 梯度 L2 归约的主机同步（阶段26）\n\n`{SOURCE}` 保留每个原生 GPU norm，仅批量回传标量。主配置三步权重、梯度、FP32 master 和实际范数均逐位一致；同轮相对CE0008吞吐+3.87%，相对原生+4.41%。这不是新Triton计算内核；更大模型、裁剪、AdamW与多卡仍未验证。\n')
    print(json.dumps({'source_id':SOURCE,'evidence_files':len(records)}))


if __name__=='__main__':main()
