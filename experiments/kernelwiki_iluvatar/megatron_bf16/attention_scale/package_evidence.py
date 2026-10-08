"""Add stage32 knowledge and small evidence, retaining every older page byte."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import verify_receipts as verifier


HERE = Path(__file__).resolve().parent
BF = HERE.parent
BUNDLE = 'bi-v150-corex42-stage32'
SOURCE_ID = 'exp-bi-v150-corex42-stage32'
PAGE = 'dense-attention-heldout-megatron-bi-v150'
EXCLUDED_DIRS = {'raw-backup', 'snapshots', '__pycache__', '.git'}
EXCLUDED_SUFFIXES = {'.pt', '.pth', '.so', '.pyc', '.tar', '.gz', '.zip', '.bin', '.partial'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill', type=Path, required=True)
    args = parser.parse_args()
    skill = args.skill.resolve()
    decision = verifier.read(HERE / 'evidence/decision.json')
    require(decision['status'] == 'completed' and decision['code_retuned'] is False
            and decision['raw_backups_complete'] is True
            and [row['model_name'] for row in decision['rounds']] == ['wide', 'deep'],
            'complete sealed wide/deep decision required')
    rounds = []
    for row in decision['rounds']:
        proof_path = HERE / 'evidence' / row['verification_path']
        require(sha(proof_path) == row['verification_sha256'], 'sealed CPU proof bytes differ')
        checked = verifier.verify(SimpleNamespace(model=row['model_name'], version='003',
                     base=HERE / 'evidence', audit_suffix=row['audit_suffix'], run_suffix=row['run_suffix'],
                     foundations_only=False, require_complete=True))
        require(checked == verifier.read(proof_path) and checked['snapshot_files_missing'] == 0,
                'all original raw bytes must still match the complete CPU proof')
        perf_path = HERE / 'evidence' / ('performance-' + row['model_name'] + '-003-' + row['run_suffix'] + '.json')
        perf = verifier.read(perf_path)
        require(sha(perf_path) == row['performance_sha256'] and perf['promoted'] is row['promoted'],
                'sealed performance decision differs')
        for metric in ('vs_control', 'vs_native'):
            require(perf[metric] == row[metric], 'sealed ratio differs')
        rounds.append((row, perf, checked, perf_path))
    require(decision['promoted_models'] == [row['model_name'] for row, _, _, _ in rounds if row['promoted']]
            and decision['promoted'] is bool(decision['promoted_models']), 'promotion scope differs')
    bundle = skill / 'evidence' / BUNDLE
    source = skill / ('sources/experiments/bi-v150-corex42-stage32.md')
    kernel = skill / ('wiki/kernels/' + PAGE + '.md')
    require(not any(path.exists() for path in (bundle, source, kernel)), 'immutable stage32 delivery already exists')
    old_manifest = skill / 'evidence/bi-v150-corex42-stage30-31/manifest.json'
    require(old_manifest.is_file(), 'preceding unchanged stage30/31 dependency bundle required')
    append_paths = (skill / 'SKILL.md', skill / 'references/iluvatar-migration-ledger.md')
    originals = {p: p.read_bytes() if p in append_paths else sha(p)
                 for p in skill.rglob('*') if p.is_file() and '.git' not in p.parts}
    selected = []
    for folder, dirs, names in os.walk(HERE):
        dirs[:] = [name for name in dirs if name not in EXCLUDED_DIRS]
        for name in names:
            path = Path(folder) / name
            require(not path.is_symlink(), 'linked file outside evidence scope')
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                continue
            require(path.stat().st_size < 16 * 1024 * 1024, 'unexpected large evidence file')
            selected.append((path, 'megatron_bf16/attention_scale/' + path.relative_to(HERE).as_posix()))
    for rel in ('evidence/cc-stage32-heldout-contract.json', 'prompts/cc-stage32-heldout-contract.txt',
                'attention/candidates/003/candidate.py', 'attention/candidates/003/generation.json'):
        selected.append((BF / rel, 'megatron_bf16/' + rel))
    bundle.mkdir()
    (bundle / '.gitattributes').write_text('* -text -eol -whitespace\n', encoding='utf-8')
    records = []
    for path, rel in sorted(selected):
        destination = bundle / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(sha(destination) == sha(path), 'small evidence bytes changed')
        records.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha(path)})
    manifest = {'schema': 'frozen-attention-stage32-additive-evidence-v1', 'source_id': SOURCE_ID,
                'asset_mode': 'verbatim', 'packager_sha256': sha(Path(__file__)), 'files': records,
                'decision_sha256': sha(HERE / 'evidence/decision.json'),
                'dependency_bundle': old_manifest.parent.relative_to(skill).as_posix(),
                'dependency_manifest_sha256': sha(old_manifest),
                'raw_policy': 'Original tensors/checkpoints and binary libraries stay in local verified backups; '
                              'the public bundle includes their hash inventories and receipts. CPU readers require '
                              'the previously shipped sibling Megatron dependencies, supplied by this project repo.',
                'existing_pages_byte_identical': True, 'base_patch_modified': False}
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (bundle / 'PROVENANCE.yaml').write_text(
        'asset_mode: verbatim\nlicense: inherits-from-source-notices\n'
        'upstream_sha: 5be9626709af2722333bf54797c954c09edeada3\n'
        'generated_by: package_evidence.py\nsource_ids: [' + SOURCE_ID + ']\nfiles:\n' +
        ''.join('  - local_path: ' + r['path'] + '\n    sha256: ' + r['sha256'] + '\n    mode: verbatim\n'
                for r in records), encoding='utf-8')
    front = ('---\nid: ' + SOURCE_ID + '\ntitle: "BI-V150 frozen Attention003 heldout validation"\n'
             'source_category: local-experiment\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
             'recorded_at: 2026-10-08\nenvironment:\n  gpu: "Iluvatar BI-V150, device0"\n'
             '  corex: "4.2.0"\n  torch: "2.4.1"\n  triton: "vendor2.1.0"\n'
             '  megatron: "5be9626709af2722333bf54797c954c09edeada3"\nevidence_files:\n' +
             ''.join('  - path: ' + p.relative_to(skill).as_posix() + '\n    sha256: ' + sha(p) + '\n'
                     for p in sorted(bundle.rglob('*')) if p.is_file()) + '---\n\n')
    body = ('# Frozen Attention003 on larger BI-V150 models\n\n'
            'The exact stage31 candidate is reused without retuning. Native and control preserve the default '
            'compiled SwiGLU, native RMSNorm and vendor causal fused softmax=True. Control is CE0008+gradnorm001; '
            'only Attention003 is added in the optimized arm. Stage31 instead includes SwiGLU004 in its control, '
            'so its13.96% gain cannot be transferred to these models.\n\n'
            '## Complete independent rounds\n\n')
    kfront = ('---\nid: kernel-' + PAGE + '\ntitle: "BI-V150 larger-model Dense Attention validation"\n'
              'type: kernel\narchitectures: [bi-v150]\ntags: [kernel-fusion, jit-compilation]\n'
              'kernel_types: [attention, fused-kernel]\nlanguages: [python, cuda-cpp]\n'
              'confidence: experimental\nreproducibility: runnable\n'
              'related: [kernel-dense-attention-megatron-bi-v150, kernel-swiglu-heldout-megatron-bi-v150]\n'
              'sources: [' + SOURCE_ID + ']\nperformance_claims:\n')
    for row, perf, proof, path in rounds:
        m = perf['model']
        shape = (f"{m['num_layers']}L H512 FFN2048 S{m['sequence_length']} V{m['vocab_size']} batch2; "
                 f"{m['parameters']} parameters; masked_softmax_fusion=True; native compiled SwiGLU")
        body += ('### ' + row['model_name'] + '\n\n' + shape + '\n\n'
                 '| Seed | Native ms/step | CE+gradnorm ms/step | +Attention003 ms/step |\n'
                 '|---|---:|---:|---:|\n')
        by_seed = {}
        for item in perf['completed_cases']:
            by_seed.setdefault(item['seed'], {})[item['arm']] = item['mean_iteration_ms']
        for seed, arms in sorted(by_seed.items()):
            body += f"|{seed}|{arms['native']:.2f}|{arms['control']:.2f}|{arms['optimized']:.2f}|\n"
        body += f"\nDecision: `{perf['decision']}`.\n\n"
        for metric, label in (('vs_control', 'Attention increment over same-round CE+gradnorm control'),
                              ('vs_native', 'Combined CE+gradnorm+Attention over same-round native')):
            ratio = perf[metric]
            body += f"- {label}: {ratio['gain_percent']:+.4f}%; all seeds faster: {ratio['all_seeds_faster']}.\n"
            kfront += ('  - gpu: "Iluvatar BI-V150"\n    dtype: bf16\n    shape: ' + json.dumps(shape) + '\n'
                       '    metric: ' + json.dumps('Native synchronized time.time throughput; ' + label + '; ' + perf['decision']) + '\n'
                       f"    value: {ratio['ratio']!r}\n    source_id: {SOURCE_ID}\n"
                       '    source_locator: ' + json.dumps('evidence/' + BUNDLE + '/megatron_bf16/attention_scale/evidence/' + path.name + '#' + metric) + '\n')
        for label, comp in proof['comparisons'].items():
            body += f"- {label}: {comp['checks']} checks/{comp['tensors']} tensors, bitwise equal={comp['all_tensor_bytes_equal']}.\n"
        body += '- Complete local original raw numerical snapshots:42; missing:0. Backup inventory:48 files.\n\n'
    body += ('## Measurement and reuse rules\n\n'
             'Three independent seeds, three rotated fresh-process arms per seed,120 steps per arm, '
             'first20 excluded; ten retained interval10 means. Primary timing is the unchanged Megatron '
             'GPU-synchronized time.time loop. No profiling/audit/checkpoint/evaluation/large backups during timing. '
             'Promotion requires strictly >1% geometric throughput versus the same-round control and every seed faster. '
             'The independent22-case/2-guard probe uses CUDA Events only as a separate microbenchmark.\n\n'
             'Correctness is a separate three-update audit, including identical initial weights/batches, per-token '
             'loss, all FP32 main_grad tensors, BF16 model parameters and FP32 SGD masters. CPU float64 tolerances '
             'remain atol3e-4/rtol1e-3. All three comparisons are bitwise equal in each shape. CPU archive verification '
             'rechecks original receipt statistics and SHA256 of locally retained bytes; it does not replay GPU arithmetic.\n\n'
             'The new Pod required restoration of exact pinned source blobs and the original dataset Makefile, '
             'then compilation with existing vendor pybind11. The initial missing-Makefile audit and missing-pybind11 '
             'build failures are retained as startup failures. No model/candidate/precision/timer gate was changed. '
             'The original600-source manifest and a separate before/after proof for the Makefile/helpers.cpp/compiled '
             'data helper are bound in CPU verification. These receipts do not seal every installed runtime binary.\n\n'
             'This validates only single BI-V150 mock-data GPT, BF16 intermediates and FP32 gradients/SGD masters, '
             'TP/PP/CP1 and dropout0. It does not establish production training, convergence, AdamW, multiGPU, '
             'FlashAttention, NVIDIA TMEM/TMA/tcgen05/WGMMA/NVFP4 equivalence or H100/B200 performance.\n\n'
             '## Pinned backend snippet\n\n```python\nimport torch\n'
             'from megatron.core.fusions.fused_softmax import ScaledUpperTriangMaskedSoftmax\n'
             '# Set S=256 or512 for the declared BF16 causal backend.\nS = 512\n'
             'scores = torch.randn(16, S, S, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
             'probs = ScaledUpperTriangMaskedSoftmax.apply(scores, 1.0)\n'
             'grad = torch.autograd.grad(probs, scores, torch.randn_like(probs))[0]\n'
             'assert grad.shape == scores.shape and grad.dtype == scores.dtype\n```\n')
    source.write_text(front + body, encoding='utf-8')
    kernel.write_text(kfront + '---\n\n' + body, encoding='utf-8')
    for path in append_paths:
        with path.open('ab') as stream:
            link = ('sources/experiments/' if path.name == 'SKILL.md' else '../sources/experiments/')
            stream.write(('\n### Stage32: larger-model Attention validation\n\n'
                'Read [stage32](' + link + 'bi-v150-corex42-stage32.md) before reusing Attention003 on '
                'larger BF16 models. It restores native compiled SwiGLU and default fused causal softmax; '
                'compare each same-round control, and enforce all-seed promotion instead of averaging away a regression.\n').encode())
    for path, old in originals.items():
        require(path.read_bytes().startswith(old) if path in append_paths else sha(path) == old,
                'older skill content changed: ' + str(path))
    print(json.dumps({'status': 'packaged', 'files': len(records), 'new_pages': 2,
                      'bytes': sum(row['bytes'] for row in records), 'old_pages_unchanged': True}))


if __name__ == '__main__':
    main()
