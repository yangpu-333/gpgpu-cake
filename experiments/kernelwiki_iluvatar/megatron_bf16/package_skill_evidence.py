"""Add evidence-backed true BF16 CE knowledge without changing NVIDIA claims."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parent
SOURCE_ID='exp-bi-v150-corex42-stage25'
BUNDLE='bi-v150-corex42-stage25'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill',type=Path,required=True)
    parser.add_argument('--decision',type=Path,required=True)
    args=parser.parse_args()
    skill=args.skill.resolve()
    decision=json.loads(args.decision.read_text())
    assert decision['status']=='completed'
    for scope,result in decision['scopes'].items():
        assert result['numerical_verification_passed'] is True
    bundle=skill/'evidence'/BUNDLE
    bundle.mkdir(exist_ok=False)
    (bundle/'.gitattributes').write_text('* -text -eol whitespace=cr-at-eol\nmegatron_bf16/evidence/** -whitespace\n')
    selected={}
    for path in ROOT.rglob('*'):
        relative=path.relative_to(ROOT)
        if not path.is_file() or any(p in relative.parts for p in ('__pycache__','snapshots')):
            continue
        if path.suffix not in ('.py','.json','.ps1','.txt','.log','.diff'):
            continue
        if path.name=='trace.json':
            continue
        selected['megatron_bf16/'+relative.as_posix()]=path
    formal=ROOT.parent/'megatron_training_entry'
    for name in ('runtime_compat.py','source_manifest.json','summarize_performance.py'):
        selected['megatron_training_entry/'+name]=formal/name
    for relative,path in selected.items():
        destination=bundle/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    files=[{'path':relative,'sha256':sha(bundle/relative),'bytes':(bundle/relative).stat().st_size}
           for relative in sorted(selected)]
    manifest={'source_id':SOURCE_ID,'recorded_at':'2026-10-03','files':files,
              'scope':'Exact scripts/contracts/raw logs/numerical receipts. Large tensor/checkpoint/Chrome trace files retained outside this skill; their hash receipts are included.'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    copied=sorted(path for path in bundle.rglob('*') if path.is_file())
    front=f'''---
id: {SOURCE_ID}
title: "BI-V150 true BF16 formal Megatron CE iteration, 2026-10-03"
source_category: local-experiment
architectures: [bi-v150]
tags: [triton, kernel-fusion, reduction, jit-compilation]
recorded_at: 2026-10-03
environment:
  gpu: "Iluvatar BI-V150, device 0"
  corex: "4.2.0"
  torch: "2.4.1"
  triton: "2.1.0 CoreX vendor build"
  megatron: "5be9626709af2722333bf54797c954c09edeada3"
  controller: "Claude Code CLI; Paratera model Claude-Opus-4.8"
evidence_files:
'''
    for path in copied:
        front+='  - path: '+path.relative_to(skill).as_posix()+'\n    sha256: '+sha(path)+'\n'
    table='| Scope | Layers / sequence / vocabulary | 0008 first two-arm | 0009 later three-arm vs native | 0009 vs same-trial0008 |\n|---|---|---:|---:|---:|\n'
    for scope,result in decision['scopes'].items():
        model=result['model']
        table+='| %s | %s / %s / %s | %.2f%% | %.2f%% | %.2f%% |\n'%(scope,model['num_layers'],model['sequence_length'],model['vocab_size'],result['candidate8_initial_gain_percent'],result['candidate9_gain_percent'],result['candidate9_vs_control_gain_percent'])
    body=f'''---

# Actual BF16 scope

This is original `pretrain_gpt.py`, native MockGPTDataset, local Torch backend,
DDP and native SGD, not a hand-written step benchmark. `--bf16` converts all
model weights (including RMSNorm weights) to BF16, without active CUDA autocast;
FP32 gradient accumulation and native FP32 master SGD parameters are retained.
Both arms share explicit old-CoreX import compatibility, unchanged pinned
600 Python-source hashes, dropout0, bias off, TP/PP/CP1. The existing dataset
Makefile, CPP and binary helper are additionally hashed for the Pod-local copy.

# Numerical failure is retained

0007 loaded BF16 logits directly in its fully fused FP32 CE forward. It passed
all504 standalone BF16 loss/dlogits/input-mutation cases, but failed five
three-step model output/gradient checks at unchanged CPUfloat64 tolerances
`atol3e-4/rtol1e-3`. Small softmax differences can alter BF16 gradient rounding
and subsequent model updates. This result rejects0007 for formal BF16 training;
standalone matrix success does not establish model-level compatibility.

0008 preserves the exact native Torch max/subtract/gather/exp/sum/log/div
forward ordering, removes only TP1 singleton all-reduce identities, and fuses
the onehot-subtract/upstream-multiply backward into Triton. 0009 additionally
stores BF16 dlogits in that backward launch while still overwriting the saved
FP32 probability buffer. The native equivalent forward remains Torch. API
counts and cached compiled-runner observations are recorded separately.

# Formal training measurements

{table}
0009 is numerically compatible in all three tested scopes, but no scope has
all three seeds faster than0008. Its small or mixed controlled gains do not
establish a stable universal performance upgrade;0008 remains the default.
The first0008 trials and later three-arm trials are separate process batches
with different storage roots after a quota recovery. Use the same-trial control
column to compare implementations, not the two historical native-relative ratios.

Every shape has a separate sealed model contract and three-update tensor audit,
including per-token FP32 loss, all FP32 main_grad, BF16 model parameters and all
FP32 master parameters; BF16 rounding cannot conceal a master update mismatch.
0009 vs0008 uses a genuine three-arm trial per seed, not historical ratios.
Each timed process runs120steps and retains the ten native log-interval10 means
ending30..120, excluding the first20steps. The original GPU-synchronized
`time.time` interval is the primary metric, including native data/schedule/DDP/
SGD. No snapshot/profile/checkpoint/eval is enabled while timing. Three seeds
are observations, not confidence intervals or production-model evidence.
See `evidence/{BUNDLE}/megatron_bf16/evidence/decision.json` and raw pair/control
records for mixed or negligible gains, candidate choice and all hashes.

# RMSNorm remains a separate BF16 task

Residual, RMSNorm input, weight and result are all BF16 in this native model.
The stage23/24 residual candidate requires FP32 residual and FP32 weight, so
it is not installed here. The vendor RMSNorm rounding probe finds that common
FP32/BF16 decompositions do not reproduce all native forward and backward
cases. A mathematical FP32 RMS formula is insufficient evidence for a native
full-BF16 replacement. The probe is diagnostic, with no fusion or speed claim.

# Reproduction

Use the recorded CoreX stack and a full clean checkout at the pinned Megatron
commit, including its native dataset build resources. From this evidence bundle,
scripts are under `megatron_bf16/`, with compatibility files in the adjacent
`megatron_training_entry/`. Use new case/output names. `run_case.py` runs native
or optimized formal training; `compare_training.py` compares three saved steps;
`verify_operator.py` checks504 BF16 CE cases. `run_performance.py` gates0008
pairs; `run_controlled.py` gates0009 three-arm trials. Contract/candidate hashes
are mandatory. Unsupported precision/shape/world size uses the adapter's native
fallback and cannot become claimed fused-call coverage.

Old NFS lock RPC and quota failures are retained as infrastructure receipts;
the final trials use the same Pod-local source for every arm, with raw snapshots
backed up separately. Kineto ChromeJSON is a diagnostic CPU/device timeline,
not ixSYS `.ptrace`, final-machine-instruction evidence or throughput timing.
No TE, real production data, TP>1, multi-card scaling or NVIDIA instruction
equivalence follows from this experiment.
'''
    source=skill/'sources/experiments/bi-v150-corex42-stage25.md'
    assert not source.exists()
    source.write_text(front+body)
    for name in ('cross-entropy-megatron-bi-v150.md','residual-rmsnorm-megatron-bi-v150.md'):
        page=skill/'wiki/kernels'/name
        text=page.read_text()
        text=re.sub(r'^sources: \[([^\n]*)\]',lambda m:'sources: ['+m.group(1)+', '+SOURCE_ID+']',text,count=1,flags=re.MULTILINE)
        if name.startswith('cross'):
            text+='''\n## True BF16 formal training, 2026-10-03

Read [stage25](../../sources/experiments/bi-v150-corex42-stage25.md) when weights
are actually BF16 outside autocast. The fully fused CE forward0007 passed504
standalone cases but failed the original three-update model gate. Keep native
Torch forward ordering for this tested path;0008 fuses backward,0009 also fuses
the final BF16 dlogits conversion. Neither establishes a fused BF16 forward.
Validate FP32 optimizer master updates separately from BF16 model parameters.
The formal interval results and larger-shape three-arm controls have their own
measurement scope and must not reuse the FP32 stage24 19.10% benchmark ratio.
'''
            selected_id=decision['scopes']['main']['preferred_candidate']
            text+=f'''\nFor the measured main BF16 scope, the selected code is
`evidence/{BUNDLE}/megatron_bf16/candidates/{selected_id}/candidate.py`.
It accepts contiguous BF16 logits[S,B,V] and int64 targets[S,B] at TP1/zero
smoothing; the guarded model adapter owns unsupported-path fallback.
'''
        else:
            text+='''\n## Full BF16 norm weights are outside the inherited fusion

[Stage25](../../sources/experiments/bi-v150-corex42-stage25.md) observes BF16
residual/input/weight/output in formal `--bf16` training. The existing fusion's
FP32-weight/residual eligibility does not cover this path. Vendor BF16 RMSNorm
rounding/backward probes do not justify replacing native math with a generic
FP32 formula. Residual/RMSNorm remains native in the new CE-only experiment.
'''
        page.write_text(text)
    ledger=skill/'references/iluvatar-migration-ledger.md'
    text=ledger.read_text()+f'''\n## 正式 BF16 训练与舍入验证（阶段25，2026-10-03）

`{SOURCE_ID}` 区分全模型 BF16 权重和 autocast；逐项核对三步 loss、全部梯度、
BF16 模型参数及 FP32 optimizer master。0007 独立504算子检查通过但完整模型失败，
保留原生前向的0008以及融合 BF16 梯度转换的0009按相同固定门槛重新验证。
三组120步正式循环计时与0008/0009同轮三臂比较另有明确范围；更大形状的收益
按实际数据报告，不套用历史19.10%。RMSNorm 保持原生，全 BF16 融合仍需独立验证。
'''
    ledger.write_text(text)
    print(json.dumps({'source_id':SOURCE_ID,'bundle_files':len(copied),'skill':str(skill)}))


if __name__=='__main__':
    main()
