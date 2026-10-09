"""Append frozen stage35 heldout knowledge after complete raw-byte verification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import verify_receipts as v

HERE = Path(__file__).resolve().parent
BF = HERE.parent
BUNDLE = 'bi-v150-corex42-stage35'
SOURCE_ID = 'exp-bi-v150-corex42-stage35'
PAGE = 'weighted-swiglu-heldout-megatron-bi-v150'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill', type=Path, required=True)
    args = parser.parse_args()
    skill = args.skill.resolve(strict=True)
    decision = v.read(HERE/'evidence/decision.json')
    v.require(decision['status']=='complete' and decision['retuned'] is False
              and decision['reader_sha256']==sha(HERE/'verify_receipts.py'), 'sealed decision required')
    proofs = {model:v.verify(model, require_raw=True) for model in ('long','experts')}
    for model, proof in proofs.items():
        v.require(proof['promoted'] and proof['comparisons']==decision['models'][model]['comparisons'],
                  'heldout numerical and performance receipts differ')
        v.require(decision['models'][model]['proof_sha256']==sha(HERE/('evidence/cpu-verify-'+model+'-raw.json')),
                  'sealed complete-byte proof differs')
    source = skill/'sources/experiments/bi-v150-corex42-stage35.md'
    kernel = skill/('wiki/kernels/'+PAGE+'.md')
    bundle = skill/'evidence'/BUNDLE
    v.require(not any(p.exists() for p in (source,kernel,bundle)), 'refusing to replace immutable stage35 evidence')
    originals = {p:sha(p) for d in ('wiki','sources','prs') for p in (skill/d).rglob('*.md')}
    v.require(len(originals)==1077, 'stage34 corpus required')
    selected = []
    for folder, dirs, names in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in ('raw-backup','results','__pycache__','.git')]
        for name in names:
            path = Path(folder)/name
            if path.suffix.lower() in ('.pt','.pth','.so','.pyc','.tar','.gz','.zip','.bin','.partial','.pstats'):
                continue
            v.require(not path.is_symlink() and path.stat().st_size<16*1024*1024, 'large or indirect public evidence')
            selected.append((path,'megatron_bf16/moe_scale/'+path.relative_to(HERE).as_posix()))
    for relative in ('evidence/cc-moe-heldout-stage35-review.json','prompts/cc-moe-heldout-stage35-review.txt'):
        selected.append((BF/relative,'megatron_bf16/'+relative))
    bundle.mkdir()
    (bundle/'.gitattributes').write_text('* -text -eol -whitespace\n',encoding='utf-8')
    rows=[]
    for path, relative in sorted(selected):
        dest=bundle/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
        v.require(sha(dest)==sha(path), 'copied original evidence differs')
        rows.append({'path':relative,'bytes':path.stat().st_size,'sha256':sha(path)})
    dependency=skill/'evidence/bi-v150-corex42-stage34/manifest.json'
    manifest={'schema':'moe-heldout-stage35-additive-evidence-v1','asset_mode':'verbatim','source_id':SOURCE_ID,
        'files':rows,'packager_sha256':sha(__file__),'decision_sha256':sha(HERE/'evidence/decision.json'),
        'dependency_bundle':dependency.parent.relative_to(skill).as_posix(),'dependency_manifest_sha256':sha(dependency),
        'existing_1077_pages_byte_identical':True,'first_four_patches_unchanged':True,
        'raw_policy':'All384 original-byte files are verified in local SHA-object/hardlink backup; public bundle retains complete inventory/receipt. CPU/CI does not repeat GPU arithmetic.'}
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (bundle/'PROVENANCE.yaml').write_text('asset_mode: verbatim\nlicense: inherits-from-source-notices\n'
        'upstream_sha: 5be9626709af2722333bf54797c954c09edeada3\n'
        'generated_by: package_evidence.py\nsource_ids: ['+SOURCE_ID+']\nfiles:\n'+
        ''.join('  - local_path: '+r['path']+'\n    sha256: '+r['sha256']+'\n    mode: verbatim\n' for r in rows),encoding='utf-8')
    source_front=('---\nid: '+SOURCE_ID+'\ntitle: "BI-V150 frozen weighted SwiGLU heldout training"\n'
        'source_category: local-experiment\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
        'recorded_at: 2026-10-09\nenvironment:\n  gpu: "Iluvatar BI-V150, device0"\n'
        '  corex: "4.2.0"\n  torch: "vendor2.4.1"\n  triton: "vendor2.1.0"\n'
        '  megatron: "5be9626709af2722333bf54797c954c09edeada3"\nevidence_files:\n'+
        ''.join('  - path: evidence/'+BUNDLE+'/'+r['path']+'\n    sha256: '+r['sha256']+'\n'
                for r in [{'path':'manifest.json','sha256':sha(bundle/'manifest.json')},*rows])+'---\n\n')
    kernel_front=('---\nid: kernel-'+PAGE+'\ntitle: "BI-V150 weighted SwiGLU on longer sequences and more experts"\n'
        'type: kernel\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
        'kernel_types: [moe, fused-kernel]\nlanguages: [python, triton]\nconfidence: experimental\n'
        'reproducibility: runnable\nrelated: [kernel-weighted-swiglu-moe-megatron-bi-v150]\n'
        'sources: ['+SOURCE_ID+']\nperformance_claims:\n')
    body=('# Frozen heldout experiment\n\n'
        'The stage34 weighted004 candidate is byte-identical (SHA256 `'+v.CANDIDATE_SHA+'`); no retuning. '
        'Claude Code CLI reads the stage34 project skill and reviews the new contracts through the configured '
        'Paratera Claude-Opus-4.8 service. Its original review and corrections are retained: E8 expands router '
        'weights by8192 parameters; parameter count is an actual native correctness/performance gate. '
        'The unproved total-check formula was replaced BEFORE freeze by complete named tensor-group coverage; '
        'all original numerical/metadata checks and CPU-float64 atol3e-4/rtol1e-3 remain unchanged.\n\n'
        'All arms use pinned native SequentialMLP, default compiled weighted SwiGLU, fused causal softmax, '
        'RMSNorm, native GEMM/router and the same exact0/1 uint8 stable-sort compatibility. Control adds '
        'CE0008+gradnorm001; optimized adds frozen weighted004 only. Dense SwiGLU004, Attention003 and '
        'GroupedGEMM are not active. Fresh preflight verifies allocation, all600 native source files and '
        '40 CPU-bool-reference sort indices. Native GPU bool sort remains unsupported on this runtime.\n\n')
    for model in ('long','experts'):
        perf=v.read(HERE/('models/'+model+'/evidence/performance-004.json'))
        contract=v.read(HERE/('models/'+model+'/contract.json'))
        config=contract['model']
        body+=(f"## {model}: S{config['sequence_length']}, E{config['num_moe_experts']}\n\n"
            f"Four layers/H512/FFN2048/B2/V1024/top-k2; {config['parameters']:,} parameters, "
            f"{contract['parameter_tensor_count']} parameter tensors.\n\n"
            '| Seed | Native ms/step | CE+gradnorm control ms/step | +weighted004 ms/step |\n'
            '|---|---:|---:|---:|\n')
        for pair in perf['pairs']:
            c=pair['cases']
            body+=f"|{pair['seed']}|{c['native']['mean_iteration_ms']:.2f}|{c['control']['mean_iteration_ms']:.2f}|{c['optimized']['mean_iteration_ms']:.2f}|\n"
        for metric,label in (('vs_control','increment over same-round CE+gradnorm control'),
                             ('vs_native','combined gain over same-round native')):
            result=perf[metric]
            body+=f"\n{label}: **{result['gain_percent']:+.4f}%**, all3 seeds faster; "
            body+=f"promotion applies to this tested configuration only ({perf['decision']}).\n"
            shape=f"L4/H512/FFN2048/S{config['sequence_length']}/B2/V1024/E{config['num_moe_experts']}/topk2"
            kernel_front+=('  - gpu: "Iluvatar BI-V150"\n    dtype: bf16\n    shape: '+json.dumps(shape)+'\n'
                '    metric: '+json.dumps('Original GPU-synchronized time.time throughput; '+label)+'\n'
                f"    value: {result['ratio']!r}\n    source_id: {SOURCE_ID}\n    source_locator: "+
                json.dumps('evidence/'+BUNDLE+'/megatron_bf16/moe_scale/models/'+model+'/evidence/performance-004.json#'+metric)+'\n')
        comparison=proofs[model]['comparisons']['comparison-control-004.json']
        body+=(f"\nEach of3 native/control/optimized comparisons passes {comparison['checks']} checks and "
               f"{comparison['tensors']} tensor comparisons, ALLBITWISE, including all parameter gradients, "
               'BF16 parameters, FP32 SGD masters, original batches/losses and12 router probability gradients. '
               f"All{config['num_layers']*3*config['num_moe_experts']} actual expert calls replay bitwise; native replay is exact first.\n")
        cv=decision['models'][model]['seed_mean_cv']
        body+=f"\nNative/control seed-mean CV: {cv['native']*100:.3f}%/{cv['control']*100:.3f}%; below the predeclared10% full-repeat trigger.\n"
    body+=('\n## Scope and verification\n\n'
        'The E4/S256 configuration changes sequence length from the stage34 shape. E8/S512 changes sequence '
        'and expert count jointly; its gain cannot be attributed to expert count alone. These are2 specific '
        'heldout shapes, not a broad scaling law. BF16/FP32-SGD masters, mock data, dropout0 and TP/PP/CP/EP1.\n\n'
        'Formal timing uses the unchanged native GPU-synchronized time.time loop:3 seeds/rotated3 arms/120 steps '
        'per process, first20 excluded,10 retained interval10 means;18 processes/2160 timed/1800 retained steps. '
        'No profiling, tensor audit, checkpoint, evaluation or large backup during timing. Strict promotion requires '
        '>1% geometric throughput over same-round control AND all3 seeds faster. Operator CUDA Event timing stays separate.\n\n'
        'Both shapes pass the original44 fixed probes+2 CPU/FP32 guards+2 actual offset1 views (mod16=2). '
        'The original cloned offset-labelled inputs do not establish pointer-offset coverage; the2 preserved-view '
        'checks supply that evidence separately. The12 ten-step printed lm_loss/router_loss strings match across '
        'all3 arms for each seed; three-update tensor equality and printed losses do not prove120-step all-tensor '
        'equivalence or convergence.\n\n'
        'Raw384 original files total22,291,588,758 bytes;227 SHA objects total9,978,190,705 bytes. '
        'The interrupted transfer and bounded resume records are retained. All bytes are backed up locally '
        'and SHA-verified; remote originals remain. Public evidence keeps the full inventory/receipt, '
        'while CPU/CI checks recorded gates, identities and statistics. CI explicitly reports unavailable '
        'large raw bytes and never reruns GPU arithmetic.\n\n'
        'No production training, AdamW, GroupedGEMM, multiGPU, SFT/RL, convergence or NVIDIA '
        'TMA/TMEM/tcgen05/WGMMA/NVFP4 equivalence is established. NVIDIA claims stay tied to their original GPUs.\n\n'
        '## Native oracle snippet\n\n```python\nimport torch\n'
        'from megatron.core.fusions.fused_bias_swiglu import weighted_bias_swiglu_impl\n'
        'x = torch.randn(257, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
        'w = torch.rand(257, 1, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
        'y = weighted_bias_swiglu_impl(x, None, w, False)\n'
        'dx, dw = torch.autograd.grad(y, (x, w), torch.randn_like(y))\n'
        'assert dx.dtype == dw.dtype == torch.bfloat16\n```\n')
    source.write_text(source_front+body,encoding='utf-8')
    kernel.write_text(kernel_front+'---\n\n'+body,encoding='utf-8')
    for path in (skill/'SKILL.md',skill/'references/iluvatar-migration-ledger.md'):
        prefix='sources/experiments/' if path.name=='SKILL.md' else '../sources/experiments/'
        with path.open('ab') as stream:
            stream.write(('\n### Stage35: Frozen MoE heldout validation\n\nRead [stage35]('+prefix+
                'bi-v150-corex42-stage35.md) for unchanged weighted004 on E4/S256 and E8/S512, '
                'same-round native/control gates and complete raw receipts. Changes in sequence and expert '
                'count are joint in the second shape; do not infer broad scaling or production gains.\n').encode())
    v.require(all(sha(path)==digest for path,digest in originals.items()), 'existing knowledge page changed')
    print(json.dumps({'status':'packaged','files':len(rows),'bytes':sum(r['bytes'] for r in rows),
                      'previous_pages_unchanged':1077,'new_pages':2}))


if __name__=='__main__':
    main()
