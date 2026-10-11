"""Append fresh BI-V150 knowledge only after both accepted rounds are sealed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import verify_completion_retry01 as completion

HERE = Path(__file__).resolve().parent
BF = HERE.parent
BUNDLE = 'bi-v150-corex42-stage35-retry01'
SOURCE_ID = 'exp-bi-v150-corex42-stage35-retry01'
PAGE = 'weighted-swiglu-heldout-megatron-bi-v150'
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill', type=Path, required=True)
    args = parser.parse_args()
    skill = args.skill.resolve(strict=True)
    decision_path = HERE / 'evidence/decision-retry01-02.json'
    decision = read(decision_path)
    assert decision['status'] == 'complete' and decision['retuned'] is False
    assert decision['completion_reader_sha256'] == sha(HERE / 'verify_completion_retry01.py')
    fresh = completion.verify(require_raw=True)
    assert fresh['original_byte_coverage_complete']
    assert fresh['models'] == decision['models'] and all(m['promoted'] for m in fresh['models'].values())
    source = skill / 'sources/experiments/bi-v150-corex42-stage35-retry01.md'
    kernel = skill / ('wiki/kernels/' + PAGE + '.md')
    bundle = skill / 'evidence' / BUNDLE
    assert not any(p.exists() for p in (source, kernel, bundle)), 'Immutable fresh release already exists'
    originals = {p: sha(p) for folder in ('wiki', 'sources', 'prs') for p in (skill / folder).rglob('*.md')}
    assert len(originals) == 1077, 'Unmodified stage34 corpus required'

    selected = {}
    def select(path):
        relative = path.relative_to(HERE).as_posix()
        assert path.is_file() and not path.is_symlink() and path.stat().st_size < 16 * 1024 * 1024
        assert path.suffix in ('.py', '.json', '.log', '.txt')
        assert not any(x in path.name.lower() for x in ('ssh', 'token', 'password'))
        selected['megatron_bf16/moe_scale_recovery01/' + relative] = path

    for path in HERE.iterdir():
        if path.suffix == '.py' or path.name in ('plan.json', 'provenance.json'):
            select(path)
    for folder in (HERE / 'models-retry01', HERE / 'evidence/retry01-metadata'):
        for directory, dirs, names in os.walk(folder):
            dirs[:] = [d for d in dirs if d != '__pycache__']
            for name in names:
                path = Path(directory) / name
                if path.suffix in ('.py', '.json', '.log'):
                    select(path)
    for path in (HERE / 'evidence').glob('*.json'):
        if not path.name.startswith('ssh-'):
            select(path)
    for path in (HERE / 'evidence/retry01-tools').glob('*.py'):
        select(path)
    for filename in ('browser_formal_retry01.py', 'browser_formal_retry01_v2.py',
                     'derive_formal_wrapper_v2.py', 'collect_formal_retry01_v2.py',
                     'collect_long_primary_diagnostic.py', 'test_resource_guard_v2.py',
                     'package_raw_transport_retry01.py', 'receive_raw_transport_retry01.py',
                     'receive_retry01_raw_http.py'):
        select(HERE / 'results' / filename)
    for relative in ('evidence/cc-moe-stage35-recovery01-review.json',
                     'prompts/cc-moe-stage35-recovery01-review.txt'):
        path = BF / relative
        selected['megatron_bf16/' + relative] = path

    bundle.mkdir()
    (bundle / '.gitattributes').write_text('* -text -eol -whitespace\n', encoding='utf-8')
    rows = []
    for relative, path in sorted(selected.items()):
        destination = bundle / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(destination) == sha(path)
        rows.append({'path': relative, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    dependency = skill / 'evidence/bi-v150-corex42-stage34/manifest.json'
    manifest = {'schema': 'fresh-heldout-stage35-additive-evidence-v2', 'asset_mode': 'verbatim',
        'source_id': SOURCE_ID, 'files': rows, 'packager_sha256': sha(__file__),
        'decision_sha256': sha(decision_path), 'dependency_bundle': dependency.parent.relative_to(skill).as_posix(),
        'dependency_manifest_sha256': sha(dependency), 'existing_1077_pages_byte_identical': True,
        'first_four_patches_unchanged': True,
        'raw_policy': 'Fresh collection: all384 original files are locally byte/SHA verified. Public bundle retains exact inventories, GPU receipts, logs and CPU readers; large tensor files stay outside Git. It does not reconstruct the lost Oct9 collection. CI explicitly reports unavailable raw bytes and does not repeat GPU arithmetic.'}
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (bundle / 'PROVENANCE.yaml').write_text('asset_mode: verbatim\nlicense: inherits-from-source-notices\n'
        'upstream_sha: 5be9626709af2722333bf54797c954c09edeada3\n'
        'generated_by: package_evidence_retry01.py\nsource_ids: [' + SOURCE_ID + ']\nfiles:\n' +
        ''.join('  - local_path: ' + r['path'] + '\n    sha256: ' + r['sha256'] + '\n    mode: verbatim\n' for r in rows), encoding='utf-8')

    source_front = ('---\nid: ' + SOURCE_ID + '\ntitle: "BI-V150 frozen weighted SwiGLU fresh heldout reproduction"\n'
        'source_category: local-experiment\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
        'recorded_at: 2026-10-11\nenvironment:\n  gpu: "Iluvatar BI-V150, device0"\n'
        '  corex: "4.2.0"\n  torch: "vendor2.4.1"\n  triton: "vendor2.1.0"\n'
        '  megatron: "5be9626709af2722333bf54797c954c09edeada3"\nevidence_files:\n' +
        ''.join('  - path: evidence/' + BUNDLE + '/' + r['path'] + '\n    sha256: ' + r['sha256'] + '\n'
                for r in [{'path': 'manifest.json', 'sha256': sha(bundle / 'manifest.json')}, *rows]) + '---\n\n')
    kernel_front = ('---\nid: kernel-' + PAGE + '\ntitle: "BI-V150 weighted SwiGLU on longer sequences and more experts"\n'
        'type: kernel\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
        'kernel_types: [moe, fused-kernel]\nlanguages: [python, triton]\nconfidence: experimental\n'
        'reproducibility: runnable\nrelated: [kernel-weighted-swiglu-moe-megatron-bi-v150]\n'
        'sources: [' + SOURCE_ID + ']\nperformance_claims:\n')
    body = ('# Frozen candidate, fresh heldout validation\n\n'
        'Use this page when considering weighted MoE SwiGLU on BI-V150/CoreX4.2. '
        'The stage34 candidate004 is unchanged (SHA256 `' + completion.CANDIDATE_SHA + '`). '
        'Cache compiled Triton callables while preserving the native BF16 intermediate rounding and router-weight gradient. '
        'The configured Claude Code CLI/Paratera Claude-Opus-4.8 review used the project skill before freeze; '
        'its original review is archived. No shape-specific retuning was performed.\n\n'
        'All arms use the pinned native SequentialMLP and default compiled weighted SwiGLU/fused causal softmax. '
        'RMSNorm, GEMM and routing remain native. The same exact0/1 uint8 stable-sort compatibility applies to all arms; '
        'native GPU bool sort is unsupported on this runtime. Control enables CE0008+gradnorm001; optimized adds weighted004 only. '
        'Dense SwiGLU004, Attention003 and GroupedGEMM are inactive. Allocation, all600 native source identities '
        'and40 CPU-reference sort-index probes passed before collection.\n\n')
    for model in ('long', 'experts'):
        result = decision['models'][model]
        primary = result['primary']
        contract = read(HERE / 'models-retry01' / model / 'contract.json')
        config = contract['model']
        perf = read(HERE / primary['performance_relative'])
        body += (f"## {model}: S{config['sequence_length']}, E{config['num_moe_experts']}\n\n"
            f"Four layers/H512/FFN2048/B2/V1024/top-k2; {config['parameters']:,} parameters, "
            f"{contract['parameter_tensor_count']} parameter tensors. Accepted round: `{primary['round']}`.\n\n"
            '| Seed | Native ms/step | CE+gradnorm control ms/step | +weighted004 ms/step |\n'
            '|---|---:|---:|---:|\n')
        for pair in perf['pairs']:
            cases = pair['cases']
            body += f"|{pair['seed']}|{cases['native']['mean_iteration_ms']:.2f}|{cases['control']['mean_iteration_ms']:.2f}|{cases['optimized']['mean_iteration_ms']:.2f}|\n"
        for metric, label in (('vs_control', 'weighted004 increment over same-round CE+gradnorm control'),
                              ('vs_native', 'combined gain over same-round native')):
            value = primary[metric]
            body += f"\n{label}: **{value['gain_percent']:+.4f}%**, all3 seeds faster.\n"
            shape = f"L4/H512/FFN2048/S{config['sequence_length']}/B2/V1024/E{config['num_moe_experts']}/topk2"
            kernel_front += ('  - gpu: "Iluvatar BI-V150"\n    dtype: bf16\n    shape: ' + json.dumps(shape) + '\n'
                '    metric: ' + json.dumps('Original GPU-synchronized time.time throughput; ' + label) + '\n'
                f"    value: {value['ratio']!r}\n    source_id: {SOURCE_ID}\n    source_locator: " +
                json.dumps('evidence/' + BUNDLE + '/megatron_bf16/moe_scale_recovery01/' + primary['performance_relative'] + '#' + metric) + '\n')
        comparison = primary['comparisons']['comparison-control-004.json']
        body += (f"\nAll3 numerical comparisons pass {comparison['checks']} checks/{comparison['tensors']} tensors each, "
            'all bitwise equal, including parameter gradients, BF16 parameters, FP32 SGD masters, batches/losses '
            'and12 router probability gradients. '
            f"All{config['num_layers'] * 3 * config['num_moe_experts']} actual expert calls replay bitwise after the native oracle passes.\n")
        cv = primary['seed_mean_cv']
        body += f"\nNative/control seed-mean CV: {cv['native']*100:.3f}%/{cv['control']*100:.3f}%. "
        body += 'An independent full repeat is required if either reaches10%; '
        body += ('the triggered repeat passed the same fixed rule.\n' if result['repeat_required'] else 'neither triggered this rule.\n')
        if result['repeat_required']:
            repeat = result['rounds'][1]
            body += f"\nRepeat increment/combined gain: {repeat['vs_control']['gain_percent']:+.4f}%/{repeat['vs_native']['gain_percent']:+.4f}%. Both rounds must pass; no selection of the faster round.\n"
    body += ('\n## Measurement and exclusions\n\n'
        'Timing is the unchanged native GPU-synchronized time.time loop:3 seeds, rotated3 arms,120 steps '
        'per process, first20 excluded and10 retained interval10 means. No profiling, tensor capture, '
        'checkpoint, evaluation or large backup runs during timing. Operator CUDA Event measurements remain separate. '
        f"Accepted rounds total{decision['accepted_formal_processes']} processes/{decision['accepted_timed_steps']} timed/"
        f"{decision['accepted_retained_steps']} retained steps. Strict promotion requires >1% geometric throughput "
        'over same-round control AND all3 seeds faster, with full repeats when predeclared CV requires them.\n\n'
        'The first complete long9-process round failed its outer post-run resource gate because a process was still present. '
        'Its ownership/state was unresolved and it later exited. Preserve that original failed receipt and exclude '
        'that round from promotion. A full independent long resource-confirmation round uses identical candidate, '
        'controllers and timing rules. The external wrapper records process state, excludes only nonexecuting zombies, '
        'retains live-compute blocking and waits5s outside timing for process exit. BothGPU and CPU boundaries '
        'must be clear before and after accepted rounds. No other jobs were stopped.\n\n'
        'Both shapes pass44 fixed probes+2 CPU/FP32 guards+2 preserved offset1 views (mod16=2). '
        'The cloned offset-labelled input alone is insufficient evidence; the preserved-view probes establish actual offset coverage. '
        'All12 ten-step printed lm_loss/router_loss strings match across arms per seed. Three-update tensor equality '
        'and printed losses do not establish120-step all-tensor equality or convergence.\n\n'
        'The E4/S256 shape changes sequence length from stage34. E8/S512 changes both sequence length and expert count; '
        'its result cannot isolate expert-count scaling. These are two tested shapes, not a scaling law. '
        'BF16/FP32-SGD masters, mock data, dropout0 and TP/PP/CP/EP1. No production training, AdamW, GroupedGEMM, '
        'multiGPU, SFT/RL or NVIDIA TMA/TMEM/tcgen05/WGMMA/NVFP4 equivalence is established.\n\n'
        'Fresh384 original files total22,291,589,014 bytes;227 SHA objects total9,978,190,961 bytes. '
        'All were backed up locally, reconstructed without tensor deserialization and SHA-verified before formal timing. '
        'Remote originals reside on persistent/private (resolved/mnt). This fresh collection does not recover the lost '
        'Oct9 raw files. Public evidence includes exact metadata/logs and full-byte receipts; CPU/CI verifies recorded '
        'gates and statistics, reports unavailable large raw bytes explicitly and does not rerun GPU arithmetic. '
        'NVIDIA knowledge and numbers retain their original architecture provenance.\n\n'
        '## Repository verification\n\n```bash\n'
        'python experiments/kernelwiki_iluvatar/megatron_bf16/moe_scale_recovery01/verify_completion_retry01.py\n'
        '# On the original local archive: append --require-raw\n```\n')
    source.write_text(source_front + body, encoding='utf-8')
    kernel.write_text(kernel_front + '---\n\n' + body, encoding='utf-8')
    for path in (skill / 'SKILL.md', skill / 'references/iluvatar-migration-ledger.md'):
        prefix = 'sources/experiments/' if path.name == 'SKILL.md' else '../sources/experiments/'
        with path.open('ab') as stream:
            stream.write(('\n### Stage35: Fresh frozen MoE heldout reproduction\n\nRead [stage35 fresh evidence](' + prefix +
                'bi-v150-corex42-stage35-retry01.md) when transferring weighted004 to E4/S256 or E8/S512. '
                'Use the accepted same-round native/control results; the rejected resource-gate round stays diagnostic. '
                'Both shapes have complete fresh raw-byte receipts. Sequence/expert changes are joint in E8/S512; '
                'do not infer broad scaling or production gains.\n').encode())
    assert all(sha(path) == digest for path, digest in originals.items()), 'Existing knowledge page changed'
    print(json.dumps({'status': 'packaged', 'files': len(rows), 'bytes': sum(r['bytes'] for r in rows),
                      'previous_pages_unchanged': 1077, 'new_pages': 2}))


if __name__ == '__main__':
    main()
