"""Package fixed held-out model measurements without changing candidate code."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--skill',type=Path,required=True);a=p.parse_args()
base=ROOT/'evidence/rmsnorm-stage28';decision=json.loads((base/'decision.json').read_text())
bundle=a.skill/'evidence/bi-v150-corex42-stage28';bundle.mkdir(exist_ok=True)
(bundle/'.gitattributes').write_text('* -text -eol -whitespace\n')
selected=[ROOT/n for n in ('run_rmsnorm_scale_audit.py','run_rmsnorm_scale_native_audit.py','run_rmsnorm_scale_performance.py','verify_rmsnorm_scale_receipts.py','backup_rmsnorm_scale.py','backup_rmsnorm_scale_native.py','recover_rmsnorm_scale_backup.py','fetch_rmsnorm_scale_evidence.py','contracts/0008-wide.json','contracts/0008-deep.json')]
selected+=list(base.rglob('*'))
records=[]
for path in sorted(set(selected)):
 rel=path.relative_to(ROOT)
 if not path.is_file() or 'snapshots' in rel.parts or '__pycache__' in rel.parts or path.suffix in ('.gz','.so'):continue
 dest=bundle/'megatron_bf16'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
 records.append({'path':'megatron_bf16/'+rel.as_posix(),'sha256':sha(dest),'bytes':dest.stat().st_size})
(bundle/'manifest.json').write_text(json.dumps({'source_id':'exp-bi-v150-corex42-stage28','files':records},indent=2)+'\n')
front="""---
id: exp-bi-v150-corex42-stage28
title: "BI-V150 held-out BF16 training shape validation"
source_category: local-experiment
architectures: [bi-v150]
tags: [reduction]
recorded_at: 2026-10-04
environment:
  gpu: "Iluvatar BI-V150 GPU0"
  corex: "4.2.0"
  torch: "2.4.1"
  megatron: "5be9626709af2722333bf54797c954c09edeada3"
evidence_files:
"""
for path in sorted(bundle.rglob('*')):
 if path.is_file():front+=f'  - path: {path.relative_to(a.skill).as_posix()}\n    sha256: {sha(path)}\n'
body="---\n\n# Fixed candidate, held-out shapes\n\n"
body+="Reuse stage27 native C++ RMSNorm006, CE0008 and gradient norm001 without tuning. Same BI-V150/CoreX4.2/Torch2.4.1, pinned original Megatron pretrain_gpt.py, true BF16 weights/FP32 gradients and SGD masters. Fixed mock data, dropout0/clipping0, TP/PP/CP1. No production or multiGPU claim.\n\n"
body+="Three-step correctness uses unchanged CPU float64 atol3e-4/rtol1e-3 for loss, all gradients, model and master parameters. Performance uses the original GPU-synchronized time.time timer: 120 steps, first20 dropped, three rotated fresh-process seeds. Native, CE+gradnorm control, added RMSNorm arms are compared in the same round. Raw logs and source hashes are bound; GPU work is serialized. Wide timing has separate deep audits interleaved between some processes, never concurrent kernels; deep timing overlaps CPU archive compression and transfer. This is not proof of a causal effect, and means are not confidence intervals.\n\n"
for model,d in decision['models'].items():
 body+=f"## {model}\n\nCorrectness: {d['checks']['passed']}/{d['checks']['total']}. Combined gain vs native: {d['vs_native']['gain_percent']:.2f}%; added RMSNorm vs CE+gradnorm: {d['vs_control']['gain_percent']:.2f}%, all seeds faster: {d['vs_control']['all_seeds_faster']}. Decision: {d['status']}.\n\n"
body+="Do not generalize main-model acceptance to these shapes solely from positive geometric means. Retain native fallback; per-shape promotion requires all three seeds faster and >1% gain vs existing optimized control. Negative and mixed results are evidence. Unchanged NVIDIA corpus and instruction claims remain GPU-specific.\n"
(a.skill/'sources/experiments/bi-v150-corex42-stage28.md').write_text(front+body)
f=a.skill/'SKILL.md'
if 'stage28 BF16 shape checks' not in f.read_text():f.write_text(f.read_text()+"\nValidate fusion on held-out shapes before promotion: [stage28 BF16 shape checks](sources/experiments/bi-v150-corex42-stage28.md). A positive mean with a regressing seed does not satisfy the fixed promotion gate.\n")
f=a.skill/'references/iluvatar-migration-ledger.md'
if 'Stage28: held-out RMSNorm shape checks' not in f.read_text():f.write_text(f.read_text()+"\n### Stage28: held-out RMSNorm shape checks\n\nReuse sealed stage27 candidate; preserve native BF16 arithmetic. Larger four/eight-layer correctness and raw three-arm training timing are separately bound. Promotion requires >1% vs control and all seeds faster. [Measured results](../sources/experiments/bi-v150-corex42-stage28.md).\n")
print(json.dumps({'source_id':'exp-bi-v150-corex42-stage28','files':len(records)}))
