"""Append stage34 knowledge and original small evidence to the stage32 skill."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import verify_receipts

HERE = Path(__file__).resolve().parent
BF = HERE.parent
BUNDLE = 'bi-v150-corex42-stage34'
SOURCE_ID = 'exp-bi-v150-corex42-stage34'
PAGE = 'weighted-swiglu-moe-megatron-bi-v150'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--skill',type=Path,required=True);p.add_argument('--version',choices=('004',),default='004');a=p.parse_args();version=a.version
    skill=a.skill.resolve(strict=True)
    proof=verify_receipts.verify(version,HERE/'evidence',True,True)
    decision=json.loads((HERE/'evidence/decision.json').read_text())
    perf=json.loads((HERE/('evidence/confirmation/performance-'+version+'.json')).read_text())
    assert decision['status']=='complete' and decision['selected_candidate']==version
    assert decision['promoted'] and decision['rounds']['004']['promoted']
    assert decision['rounds']['004-confirmation']['promoted'] is proof['promoted'] is perf['promoted'] is True
    source=skill/'sources/experiments/bi-v150-corex42-stage34.md'
    kernel=skill/('wiki/kernels/'+PAGE+'.md');bundle=skill/'evidence'/BUNDLE
    assert not any(p.exists() for p in (source,kernel,bundle))
    originals={p:sha(p) for d in ('wiki','sources','prs') for p in (skill/d).rglob('*.md')}
    assert len(originals)==1075
    selected=[]
    for folder,dirs,names in os.walk(HERE):
        dirs[:]=[d for d in dirs if d not in ('raw-backup','results','__pycache__','.git')]
        for name in names:
            path=Path(folder)/name
            if path.suffix.lower() in ('.pt','.pth','.so','.pyc','.tar','.gz','.zip','.bin','.partial','.pstats'):continue
            assert not path.is_symlink() and path.stat().st_size<16*1024*1024
            selected.append((path,'megatron_bf16/moe_weighted/'+path.relative_to(HERE).as_posix()))
    for folder,dirs,names in os.walk(BF/'moe'):
        dirs[:]=[d for d in dirs if d not in ('raw-backup','results','__pycache__','.git')]
        for name in names:
            path=Path(folder)/name
            if path.suffix.lower() in ('.pt','.pth','.so','.pyc','.tar','.gz','.zip','.bin','.partial','.pstats'):continue
            assert path.stat().st_size<16*1024*1024 and not path.is_symlink()
            selected.append((path,'megatron_bf16/moe/'+path.relative_to(BF/'moe').as_posix()))
    for rel in ('evidence/cc-moe-native-preflight-review.json','prompts/cc-moe-native-preflight-review.txt'):
        path=BF/rel
        if path.is_file():selected.append((path,'megatron_bf16/'+rel))
    for v in ('001','002','003','004'):
        for rel in ('evidence/cc-moe-weighted-'+v+'.json','prompts/cc-moe-weighted-'+v+'.txt'):
            selected.append((BF/rel,'megatron_bf16/'+rel))
    bundle.mkdir();(bundle/'.gitattributes').write_text('* -text -eol -whitespace\n',encoding='utf-8')
    rows=[]
    for path,rel in sorted(selected):
        dest=bundle/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
        assert sha(dest)==sha(path);rows.append({'path':rel,'bytes':path.stat().st_size,'sha256':sha(path)})
    dep=skill/'evidence/bi-v150-corex42-stage32/manifest.json'
    manifest={'schema':'moe-weighted-stage34-additive-evidence-v1','asset_mode':'verbatim','source_id':SOURCE_ID,
        'files':rows,'packager_sha256':sha(__file__),'decision_sha256':sha(HERE/'evidence/decision.json'),
        'dependency_bundle':dep.parent.relative_to(skill).as_posix(),'dependency_manifest_sha256':sha(dep),
        'existing_1075_pages_byte_identical':True,'first_three_patches_unchanged':True,
        'raw_policy':'Complete original tensor/checkpoint bytes stay in verified local backup; public bundle retains their full hash inventory and receipt. No arithmetic replay in CPU/CI.'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (bundle/'PROVENANCE.yaml').write_text('asset_mode: verbatim\nlicense: inherits-from-source-notices\n'
        'upstream_sha: 5be9626709af2722333bf54797c954c09edeada3\n'
        'generated_by: package_evidence.py\nsource_ids: ['+SOURCE_ID+']\nfiles:\n'+
        ''.join('  - local_path: '+r['path']+'\n    sha256: '+r['sha256']+'\n    mode: verbatim\n' for r in rows),encoding='utf-8')
    front=('---\nid: '+SOURCE_ID+'\ntitle: "BI-V150 weighted SwiGLU in native MoE training"\n'
        'source_category: local-experiment\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
        'recorded_at: 2026-10-09\nenvironment:\n  gpu: "Iluvatar BI-V150, device0"\n  corex: "4.2.0"\n'
        '  torch: "vendor2.4.1"\n  triton: "vendor2.1.0"\n  megatron: "5be9626709af2722333bf54797c954c09edeada3"\nevidence_files:\n'+
        ''.join('  - path: '+p.relative_to(skill).as_posix()+'\n    sha256: '+sha(p)+'\n' for p in sorted(bundle.rglob('*')) if p.is_file())+'---\n\n')
    shape='4L H512 FFN2048 S128 V1024 batch2 experts4 topk2 SequentialMLP; 55,652,864 parameters'
    body=('# Weighted SwiGLU in BI-V150 MoE training\n\n'+shape+'\n\n'
        'Native is pinned local Torch Megatron with default compiled weighted SwiGLU, native RMSNorm and fused causal softmax=True. '
        'Control adds only CE0008 and gradnorm001; optimized adds weighted SwiGLU'+version+'. Dense SwiGLU004, Attention003, '
        'GroupedGEMM and candidate router/GEMM kernels are absent. The explicit common bool-to-exact0/1-uint8 stable-sort '
        'compatibility was admitted by 32 CPU-reference index probes in stage33. Original GPU bool stable sort is unavailable; '
        'all three arms use the same compatibility.\n\n'
        'BF16 model/intermediates, FP32 main_grad and SGD master, mock data, dropout0, no autocast, TP/PP/CP/EP1. '
        'Actual weighted input is BF16[N,4096], probabilities BF16[N,1], bias=None, fp8_store=False; expert token counts 38–256 '
        'in the three-update native capture. Auxiliary router-loss coefficient 0.01 is retained.\n\n'
        '## Full three-arm timing\n\n| Seed | Native ms/step | CE+gradnorm ms/step | +weighted SwiGLU'+version+' ms/step |\n'
        '|---|---:|---:|---:|\n')
    for pair in perf['pairs']:
        c=pair['cases'];body+=f"|{pair['seed']}|{c['native']['mean_iteration_ms']:.2f}|{c['control']['mean_iteration_ms']:.2f}|{c['optimized']['mean_iteration_ms']:.2f}|\n"
    body+=f"\nDecision after both independent rounds: `{decision['decision']}`.\n\n"
    kfront=('---\nid: kernel-'+PAGE+'\ntitle: "BI-V150 native MoE weighted SwiGLU optimization"\n'
        'type: kernel\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\nkernel_types: [moe, fused-kernel]\n'
        'languages: [python, triton]\nconfidence: experimental\nreproducibility: runnable\n'
        'related: [kernel-swiglu-megatron-bi-v150, kernel-dense-attention-heldout-megatron-bi-v150]\n'
        'sources: ['+SOURCE_ID+']\nperformance_claims:\n')
    for metric,label in [('vs_control','Weighted activation increment over same-round CE+gradnorm control'),
                         ('vs_native','Combined CE+gradnorm+weighted activation over same-round native')]:
        r=perf[metric];body+=f"- {label}: {r['gain_percent']:+.4f}%; all seeds faster: {r['all_seeds_faster']}.\n"
        kfront+=('  - gpu: "Iluvatar BI-V150"\n    dtype: bf16\n    shape: '+json.dumps(shape)+'\n'
            '    metric: '+json.dumps('Original synchronized time.time throughput; '+label+'; '+perf['decision'])+'\n'
            f"    value: {r['ratio']!r}\n    source_id: {SOURCE_ID}\n    source_locator: "+
            json.dumps('evidence/'+BUNDLE+'/megatron_bf16/moe_weighted/evidence/confirmation/performance-'+version+'.json#'+metric)+'\n')
    first=json.loads((HERE/('evidence/performance-'+version+'.json')).read_text())
    body+=('\nThe first complete round measured '+f"{first['vs_control']['gain_percent']:+.4f}% versus control and {first['vs_native']['gain_percent']:+.4f}% versus native. "
        'Some native/control arms had larger variance (native73.77/86.03/100.71ms; control93.94/75.45/71.92ms), '
        'so an independent complete nine-process repeat used IDENTICAL controller/candidate bytes and the same 3 seeds/orders/120-step timer. '
        'The table above is that repeat; both original rounds remain archived. Promotion requires both rounds to pass the original >1%/all3-faster rule.\n')
    pc=decision['profile_counts_diagnostic_only']
    body+=('\nSeparate 20-step profiles (not formal timing) record '+
        f"{pc['host-profile-optimized-003']['triton_compile_calls']} Triton compile calls for 003 and {pc['host-profile-optimized-004']['triton_compile_calls']} for 004. "
        'Original cProfile data is retained locally and hash-bound; the public bundle includes original parsed counters and receipts.\n')
    body+=('\n## Numerical boundaries and failed iterations\n\n'
        'The fixed independent probe has 44 cases (empty, unit, probability/signed weights, widths257/769/2048, '
        'actual expert replay, fresh inputs and nondefault streams) plus 2 CPU/FP32 guards. The main probe clones its inputs; '
        'its offset-labelled rows do not establish actual pointer-offset coverage. A separate preserved-view probe adds 2 '
        'genuine contiguous offset1/mod16=2 cases, without altering the original probe or tolerance. All 48 original expert '
        'calls are replayed against their captured upstream gradients; native replay must be bitwise before candidate comparison.\n\n'
        'Full three-update native/control/optimized comparisons each contain 2627 checks and 1289 tensors, including all 56 '
        'FP32 gradient/master sets, BF16 parameters, original batches/losses,12 router maps/probabilities and 12 probability '
        'gradients. Tolerances stay CPU-float64 atol3e-4/rtol1e-3. Candidate'+version+' passes the complete scope; bitwise status '
        'is recorded in the CPU proof. Three-update equality does not prove 120-step tensor equivalence or convergence.\n\n'
        'Candidate001 passed the small probe but failed 2 step 3 embedding-gradient tensors (8 elements outside tolerance). '
        'All 48 native replays isolate one weights_grad BF16 rounding boundary at call 47 N97 (abs9.5367e-7); output and dx '
        'are bitwise on all 48. Candidate002 replaced this reduction with a standalone torch.compile function and passed '
        'training bitwise, but full nine-process timing regressed: control71–73ms versus candidate157–784ms/step. '
        'It is not promoted; original records are retained. Candidate003 mirrors the retained original Inductor two-dimensional '
        'accumulator loop (XBLOCK1/RBLOCK2048/8warps/1stage forH2048) and restores immutable compiled-launch closure reuse '
        'with current runtime arguments and streams, but its exact N/grid keys still trigger vendor shape-specific compilation. '
        'Its complete three-seed training also regresses (optimized162–783ms/step versus control70–73ms/step), and it is not promoted. '
        'An independent 20-step host profile observes 220 Triton compile calls in 268 launcher calls. The installed vendor JIT '
        'includes tensor shapes in its cache key. Candidate004 instead caches immutable CompiledKernel objects across dynamic '
        'N and grids, keeps runtime sizes unspecialized, and launches each binary with the current grid/tensors/stream. '
        'Only compiled binaries are cached, never input/output tensors; unsupported inputs preserve the native callback. The captured '
        'compiler-cache catalog contains both original native and candidate002-generated entries; the selected original '
        'weighted-backward wrapper and reduction are separately hash-bound, not part of the 600 original source manifest.\n\n'
        'Primary timing remains the unchanged native GPU-synchronized time.time loop: two complete rounds, each 3 seeds/3 rotated arms/120 steps per '
        'process, first 20 excluded, ten retained interval-10 means. No profiling, tensor audit, checkpoint, evaluation or '
        'large backups during timing. Promotion requires >1% geometric throughput versus same-round control and every '
        'seed faster. CUDA Event operator numbers include host/autograd submission gaps and are kept separately. '
        'CPU/CI verifies raw receipt coverage/hashes/statistics, never reruns GPU arithmetic; unavailable large raw bytes '
        'remain explicit in CI. Complete local raw inventory also preserves failed-candidate and diagnostic snapshots.\n\n'
        'This does not validate production training, AdamW, GroupedGEMM, multiGPU, RL/SFT, convergence, or NVIDIA '
        'TMEM/TMA/tcgen05/WGMMA/NVFP4 instruction equivalents. Do not transfer NVIDIA performance claims to BI-V150.\n\n'
        '## Pinned native oracle snippet\n\n```python\nimport torch\n'
        'from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl\n'
        'x = torch.randn(97, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
        'w = torch.rand(97, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
        'y = weighted_bias_swiglu_impl(x, None, w, False)\n'
        'dx, dw = torch.autograd.grad(y, (x, w), torch.randn_like(y))\n'
        'assert dx.dtype == dw.dtype == torch.bfloat16\n```\n')
    source.write_text(front+body,encoding='utf-8');kernel.write_text(kfront+'---\n\n'+body,encoding='utf-8')
    for file in (skill/'SKILL.md',skill/'references/iluvatar-migration-ledger.md'):
        prefix='sources/experiments/' if file.name=='SKILL.md' else '../sources/experiments/'
        with file.open('ab') as stream:
            stream.write(('\n### Stage34: MoE weighted SwiGLU\n\nRead [stage34]('+prefix+'bi-v150-corex42-stage34.md) '
                'for actual router-weight gradients, reduction rounding boundaries and full same-round training gates. '
                'A passing operator microbenchmark does not establish full training correctness.\n').encode())
    assert all(sha(path)==digest for path,digest in originals.items())
    print(json.dumps({'status':'packaged','files':len(rows),'bytes':sum(r['bytes'] for r in rows),
                      'previous_pages_unchanged':1075,'new_pages':2}))


if __name__=='__main__':
    main()
