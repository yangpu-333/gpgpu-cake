"""Package stage27 measured BF16 native-backend fusion, including rejected trials."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent;SOURCE='exp-bi-v150-corex42-stage27';BUNDLE='bi-v150-corex42-stage27'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--skill',type=Path,required=True);a=p.parse_args()
decision=json.loads((ROOT/'evidence/rmsnorm-stage27/decision.json').read_text());assert decision['status']=='accepted_for_tested_main_scope'
bundle=a.skill/'evidence'/BUNDLE;bundle.mkdir(exist_ok=False);(bundle/'.gitattributes').write_text('* -text -eol -whitespace\n')
selected=[]
for name in ('probe_rmsnorm_semantics.py','probe_rmsnorm_candidate.py','probe_rmsnorm_layouts.py','probe_rmsnorm_backward_order.py','probe_compiled_launch.py',
             'launch_rmsnorm.py','run_rmsnorm_audit.py','run_rmsnorm_performance.py','build_rmsnorm_cpp.py','backup_rmsnorm.py','verify_rmsnorm_receipts.py',
             'launch_training.py','kernel_adapter.py','run_case.py','compare_training.py','contracts/0008-main.json','candidates/0008/candidate.py','gradnorm_candidates/001/candidate.py'):
 selected.append(ROOT/name)
selected+=list((ROOT/'rmsnorm_candidates').rglob('*'))+list((ROOT/'evidence/rmsnorm-stage27').rglob('*'))
selected+=list((ROOT/'prompts').glob('cc-rmsnorm-*.txt'))+list((ROOT/'evidence').glob('*rmsnorm*.json'))
records=[]
for path in sorted(set(selected)):
 rel=path.relative_to(ROOT)
 if not path.is_file() or 'snapshots' in rel.parts or '__pycache__' in rel.parts or path.suffix in ('.gz','.so'):continue
 dest=bundle/'megatron_bf16'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 records.append({'path':'megatron_bf16/'+rel.as_posix(),'sha256':sha(dest),'bytes':dest.stat().st_size})
for name in ('runtime_compat.py','source_manifest.json','summarize_performance.py'):
 dest=bundle/'megatron_training_entry'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT.parent/'megatron_training_entry'/name,dest)
 records.append({'path':'megatron_training_entry/'+name,'sha256':sha(dest),'bytes':dest.stat().st_size})
(bundle/'manifest.json').write_text(json.dumps({'source_id':SOURCE,'files':records},indent=2)+'\n')
front=f'''---
id: {SOURCE}
title: "BI-V150 formal BF16 RMSNorm native C++ pointwise fusion"
source_category: local-experiment
architectures: [bi-v150]
tags: [reduction]
recorded_at: 2026-10-03
environment:
  gpu: "Iluvatar BI-V150, device 0"
  corex: "4.2.0"
  torch: "2.4.1"
  triton: "vendor 2.1.0; rejected/slow candidates retained"
  megatron: "5be9626709af2722333bf54797c954c09edeada3"
  controller: "Claude Code CLI / Paratera Claude-Opus-4.8"
evidence_files:
'''
for path in sorted(bundle.rglob('*')):
 if path.is_file():front+=f'  - path: {path.relative_to(a.skill).as_posix()}\n    sha256: {sha(path)}\n'
body=f'''---

# Preserve native BF16 arithmetic and the gradient graph

The tested vendor Torch2.4.1 RMSNorm uses BF16 intermediate arithmetic in
pow/mean/epsilon/rsqrt and both affine products. Direct FP32 formulas did not
match the recorded native output. Native BF16 pow(3) also rounds between
its two products. Do not assume standard FP32 opmath semantics from another
backend or NVIDIA kernel. The source and independent CPU checks are retained.

Candidate006 uses native C++/CUDA pointwise affine kernels on CoreX4.2. It keeps
native Torch pow/mean/add_/rsqrt and their autograd branch outside the custom
affine Function, with input edge order x,rstd,weight. Forward explicitly rounds
BF16 x*rstd before multiplying weight; backward fuses products with the same
BF16 round points, then calls the original Torch sum_to_size reductions.
It does not fuse residual addition or the normalization reduction. C++ dispatch
avoids the tested Python/Triton overhead; CUDA compatibility labels do not prove
NVIDIA hardware instruction equivalence. SDK CUDA10.2 and arch7.0 are compiler
compatibility values, not the actual BI-V150 hardware architecture.

The tested Python shim SHA is `{decision['candidate_sha256']}`; loaded library
SHA is `{decision['artifact_sha256']}`. Each formal process binds the library,
C++/CUDA sources, wrapper, unchanged600native sources and CE/gradnorm parents.
Model-generated Windows files and deployed Linux files have identical text but
different EOL serialization; receipts bind actual tested Linux bytes.

# Full training validation

Single BI-V150, main model4layers/H512/FFN2048/heads8/S128/V1024/microbatch2,
TP/PP/CP1, trueBF16 weights, FP32 gradient accumulation and FP32 SGD masters,
no autocast/clipping/dropout, native MockGPTDataset/NullTokenizer, localTorch.
Both optimized arms use CE0008 and gradient-norm001. Three-update comparison
passed1475/1475 gates; all637compared loss/gradient/model/master tensors were
bitwise equal. Twelve scaled standalone cases plus12fresh-input three-dimensional
and strided-layout cases passed72output/dx/dw gates. Copies are compared on CPU
float64 at unchanged atol3e-4/rtol1e-3; vendor GPUfloat64 comparisons are not used.
Standalone gates are not a substitute for formal residual-graph training checks.

# Formal synchronized throughput

Nine fresh120step processes rotate native, CE+gradnorm control, and addedRMSNorm
orders across three seeds. Original GPU-synchronized time.time intervals are
retained; first20steps dropped,10complete interval means ending30..120 averaged.
No tensor audit/profile/evaluation/checkpoint during timed runs. Each optimized
process has1080RMSNorm calls and zero fallback. These are API coverage records,
not a separate profiler claim about every individual compiled-kernel launch.

| Seed | Native ms | CE+gradnorm ms | Added RMSNorm ms |
|---|---:|---:|---:|
|27202|31.47|30.54|29.31|
|27203|31.51|29.68|29.47|
|27204|31.44|29.01|28.66|

| gpu | dtype | shape | metric | value | source_id |
|---|---|---|---|---|---|
|BI-V150 GPU0|BF16 weights/FP32 gradients and master|4L H512 S128 V1024 batch2|formal-loop throughput vs same-round CE+gradnorm|1.020319x (+2.03%)|{SOURCE}|
|BI-V150 GPU0|BF16 weights/FP32 gradients and master|4L H512 S128 V1024 batch2|formal-loop throughput vs same-round original native|1.079904x (+7.99%)|{SOURCE}|

All three seeds faster for both comparisons; three observations are not a
confidence interval. Do not add or multiply percentages from older trial batches
or compare standalone synchronized perf_counter with formal step timing.

# Rejected trials and limits

Five Triton iterations preceded the native C++ candidate. Candidate003 passed
full1475gate training validation but lost2.81% vs same-round CE+gradnorm control,
so was rejected. Candidate004 failed standalone dx; correcting native pow3
rounding yielded005 with standalone success but8failed full-training gates.
Changing residual-branch BF16 accumulation association is a hypothesis, not an
established causal finding. The accepted candidate retains003's native branch
structure. Earlier incomplete snapshot metadata and missing case metadata are
retained as failures and excluded from accepted throughput/correctness proof.

This is not production data, convergence, multiGPU, AdamW, nonfinite/clipping
compatibility, TE/Apex performance, or a universal RMSNorm replacement. Only
the tested main configuration is accepted opt-in. SDK/ninja build logs and
source hashes are recorded; isolated native build does not modify Torch/CoreX
or pinned model sources. Raw snapshots and the exact binary are archived
outsideGit; hash receipts included here. Preserve the existing native fallback.

# Evidence entry points

Read `megatron_bf16/evidence/rmsnorm-stage27/decision.json`,
`comparison-006-complete.json`, `performance-006.json` and
`cpp-build-006-receipt.json` inside this bundle. The canonical compiled sources
are `affine-006.cpp` and `affine-006.cu` next to those receipts. Controller build
script and model output records are included; ABI/platform rebuild is required
before importing the candidate shim. CPU-only replay:

```bash
python evidence/{BUNDLE}/megatron_bf16/verify_rmsnorm_receipts.py
```
'''
(a.skill/'sources/experiments'/f'{BUNDLE}.md').write_text(front+body)
skill=a.skill/'SKILL.md';text=skill.read_text();text+='\nFor native BF16 RMSNorm, preserve staged rounding and residual-gradient graph structure; see [stage27 native pointwise fusion](sources/experiments/bi-v150-corex42-stage27.md). Code-only compiled launch caching and C++ dispatch must be measured separately from GPU arithmetic; never cache test results.\n';skill.write_text(text)
ledger=a.skill/'references/iluvatar-migration-ledger.md';text=ledger.read_text();text+='\n### Stage27: native BF16 RMSNorm affine fusion\n\nPreserve vendor BF16 rounding and original normalization gradient branch. Six Claude Code iterations; native C++/CUDA affine pointwise candidate006 passes1475full-training gates with637bitwise-equal tensors. Same-round formal main-model throughput +2.03% vs CE+gradnorm, +7.99% vs native. Candidate003 -2.81% vs control; candidate005 standalone success but8full-training gates fail. Scope single BI-V150/main BF16/SGD/no clipping/mock data; no TE/Apex or NVIDIA instruction equivalence. [Evidence](../sources/experiments/bi-v150-corex42-stage27.md).\n';ledger.write_text(text)
print(json.dumps({'source_id':SOURCE,'evidence_files':len(records)}))
