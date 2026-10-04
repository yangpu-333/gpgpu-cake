"""Package stage29 SwiGLU evidence without publishing raw tensors or binaries.

Run only after the root controller seals evidence/swiglu-stage29/decision.json.
This adds two new BI-V150 pages and appends guidance to SKILL and the migration
ledger. Existing architecture pages and earlier evidence bundles are untouched.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil


SCRIPT_ROOT = Path(__file__).resolve().parent
BF = SCRIPT_ROOT.parent
RAW = BF / 'evidence/swiglu-stage29'
SOURCE = 'exp-bi-v150-corex42-stage29'
BUNDLE = 'bi-v150-corex42-stage29'
KERNEL_ID = 'kernel-swiglu-megatron-bi-v150'
EXCLUDED_SUFFIXES = {'.pt', '.pth', '.so', '.dll', '.pyd', '.a', '.o', '.obj',
                     '.pyc', '.gz', '.zip', '.tar', '.7z', '.bin', '.safetensors',
                     '.cubin', '.fatbin', '.bc'}
EXCLUDED_DIRECTORIES = {'snapshots', '__pycache__', 'deps', '.objects', '.object-store'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def yaml_scalar(value):
    return json.dumps(str(value), ensure_ascii=False)


def collect_files():
    selected = set(SCRIPT_ROOT.rglob('*')) | set(RAW.rglob('*'))
    selected |= set((BF / 'prompts').glob('cc-swiglu-*.txt'))
    selected |= set((BF / 'evidence').glob('cc-swiglu-*.json'))
    selected |= {BF / name for name in
                 ('launch_training.py', 'kernel_adapter.py', 'run_case.py',
                  'compare_training.py', 'summarize_performance.py', 'invoke_cc.ps1',
                  'evidence/gradnorm-stage26/launch_gradnorm-04.py',
                  'candidates/0008/candidate.py', 'gradnorm_candidates/001/candidate.py')}
    files = []
    omitted = []
    for path in sorted(selected):
        if not path.is_file():
            continue
        rel = path.relative_to(BF)
        excluded = (bool(EXCLUDED_DIRECTORIES.intersection(rel.parts))
                    or any(part.startswith('cpp-build-') for part in rel.parts[:-1])
                    or path.suffix.lower() in EXCLUDED_SUFFIXES)
        if excluded:
            omitted.append({'path': rel.as_posix(), 'bytes': path.stat().st_size})
        else:
            files.append((path, 'megatron_bf16/' + rel.as_posix()))
    for name in ('runtime_compat.py', 'source_manifest.json', 'summarize_performance.py'):
        path = BF.parent / 'megatron_training_entry' / name
        assert path.is_file(), 'missing frozen compatibility/source file: ' + str(path)
        files.append((path, 'megatron_training_entry/' + name))
    return files, omitted


def summarize_performance(path):
    report = read(path)
    if report.get('status') != 'validated':
        return None
    by_seed = {}
    for case in report['completed_cases']:
        assert case['arm'] in ('native', 'control', 'optimized')
        rows = by_seed.setdefault(int(case['seed']), {})
        assert case['arm'] not in rows, 'duplicate timing arm/seed'
        value = float(case['mean_iteration_ms'])
        assert math.isfinite(value) and value > 0
        rows[case['arm']] = value
    assert len(by_seed) == 3 and set(by_seed) == set(report['seeds'])
    assert all(set(row) == {'native', 'control', 'optimized'} for row in by_seed.values())
    model = report['model']
    assert model['parameters'] == 17895936 and model['gated_linear_unit'] is True
    ratios = {}
    for label, left, right in (('vs_native', 'native', 'optimized'),
                               ('vs_control', 'control', 'optimized'),
                               ('control_vs_native', 'native', 'control')):
        values = [row[left] / row[right] for row in by_seed.values()]
        ratio = math.exp(sum(math.log(value) for value in values) / len(values))
        expected = report[label]
        assert math.isclose(ratio, float(expected['ratio']), rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose((ratio-1)*100, float(expected['gain_percent']),
                            rel_tol=1e-12, abs_tol=1e-10)
        assert all(value > 1 for value in values) == expected['all_seeds_faster']
        ratios[label] = {'ratio': ratio, 'gain_percent': (ratio-1)*100,
                         'all_seeds_faster': all(value > 1 for value in values)}
    promoted = bool(report['promoted'])
    if promoted:
        assert ratios['vs_control']['ratio'] > 1.01
        assert ratios['vs_control']['all_seeds_faster']
    return {'version': report['version'], 'file': path.name, 'sha256': sha(path),
            'by_seed': by_seed, 'ratios': ratios, 'promoted': promoted,
            'decision': report['decision'], 'scope': report['scope'], 'model': model}


def performance_text(rounds):
    if not rounds:
        return ('No validated formal throughput report is present. Standalone Event '
                'measurements and compiler/controller failures are not training throughput.\n')
    sections = []
    for trial in rounds:
        section = ('### Candidate ' + trial['version'] + ' — ' + trial['file'] + '\n\n'
                   '| Seed | Native ms/step | CE0008 + gradnorm001 control | Control + SwiGLU candidate |\n'
                   '|---|---:|---:|---:|\n')
        for seed, row in sorted(trial['by_seed'].items()):
            section += (f"|{seed}|{row['native']:.2f}|{row['control']:.2f}|"
                        f"{row['optimized']:.2f}|\n")
        section += '\n'
        for label, description in (
                ('vs_control', 'Added SwiGLU versus the same-round CE + gradnorm control'),
                ('vs_native', 'Combined CE + gradnorm + SwiGLU versus the same-round original native'),
                ('control_vs_native', 'CE + gradnorm control versus the same-round original native')):
            ratio = trial['ratios'][label]
            section += (f"- {description}: {ratio['ratio']:.8f}x "
                        f"({ratio['gain_percent']:+.2f}%); every seed faster: "
                        f"{str(ratio['all_seeds_faster']).lower()}.\n")
        section += (f"\nRound decision: `{trial['decision']}`; promoted: "
                    f"`{str(trial['promoted']).lower()}`. A positive combined ratio does not "
                    'establish that the new activation improves the existing control.\n')
        sections.append(section)
    return '\n'.join(sections)


def numerical_text():
    sections = []
    for path in sorted(RAW.glob('audit-summary-main-*.json')):
        report = read(path)
        sections.append(f"- `{path.name}`: full audit passed: `{report['passed']}`.")
        for name, comparison in report['comparisons'].items():
            checks = comparison['check_summary']
            sections.append(f"  - {name}: {checks['passed']}/{checks['total']} gates; "
                            f"{comparison['tensor_comparisons']} compared tensors; "
                            f"all bitwise equal: `{comparison['all_tensors_bitwise_equal']}`.")
    for path in sorted(RAW.glob('probe-*.json')):
        report = read(path)
        if not isinstance(report.get('checks'), list):
            continue
        rows = report['checks']
        failures = sum(not (row.get('input_unchanged', True)
                           and row.get('upstream_unchanged', True)
                           and all(check.get('passed') is True for check in row.get('checks', [])))
                       for row in rows)
        sections.append(f"- `{path.name}`: `{report['status']}`, {failures}/{len(rows)} "
                        'numerical cases rejected. A case can contain multiple output/gradient gates.')
    return '\n'.join(sections) + '\n'


def failure_text():
    sections = []
    receipt = RAW / 'cpp-build-002-receipt.json'
    log = RAW / 'build-002.stdout.log'
    if receipt.exists() and read(receipt).get('status') == 'failed':
        sections.append('- Candidate002: the first explicit build failed before numerical '
                        'evaluation. The controller supplied NVIDIA-style `--fmad=false`; '
                        'the CoreX clang compiler rejected that flag. Preserve the failed '
                        'receipt/log separately from later controller flags and candidate code.')
        assert log.exists() and '--fmad=false' in log.read_text(encoding='utf-8', errors='replace')
    lazy_log = RAW / 'probe-003-v2.stderr.log'
    if lazy_log.exists():
        log_text = lazy_log.read_text(encoding='utf-8', errors='replace')
        assert 'NoneType' in log_text and '__file__' in log_text
        sections.append('- Candidate003 initially hit a controller compatibility failure: '
                        'probe-v2 read `_ext.__file__` before the candidate\'s lazy loader ran. '
                        'That failure is not a numerical result. Probe-v3 loads and hash-binds '
                        'the library before its fixed cases; the original probe-v2 bytes and '
                        'stderr remain evidence.')
    sections.append('- Do not report throughput for candidates that stopped at compiler, '
                    'controller or numerical gates. No full-training measurement is inferred '
                    'from a passing standalone formula or an Event microbenchmark.')
    return '\n'.join(sections) + '\n'


def source_front(skill, bundle, date):
    text = (f'---\nid: {SOURCE}\n'
            'title: "BI-V150 native default compiled SwiGLU iteration"\n'
            'source_category: local-experiment\narchitectures: [bi-v150]\n'
            'tags: [triton, kernel-fusion, gated-dual-gemm, jit-compilation]\n'
            f'recorded_at: {date}\n'
            'environment:\n  gpu: "Iluvatar BI-V150, device 0"\n'
            '  corex: "4.2.0"\n  torch: "2.4.1"\n'
            '  triton: "vendor 2.1.0"\n'
            '  megatron: "5be9626709af2722333bf54797c954c09edeada3"\n'
            '  controller: "Claude Code CLI / Paratera Claude-Opus-4.8"\n'
            'evidence_files:\n')
    for path in sorted(bundle.rglob('*')):
        if path.is_file():
            text += (f'  - path: {path.relative_to(skill).as_posix()}\n'
                     f'    sha256: {sha(path)}\n')
    return text + '---\n\n'


def kernel_front(rounds):
    text = (f'---\nid: {KERNEL_ID}\n'
            'title: "Native Megatron SwiGLU activation on BI-V150"\n'
            'type: kernel\narchitectures: [bi-v150]\n'
            'tags: [gated-dual-gemm, kernel-fusion, fused-kernel, triton, jit-compilation]\n'
            'kernel_types: [elementwise, fused-kernel]\n'
            'languages: [triton, cuda-cpp]\nconfidence: experimental\n'
            'reproducibility: runnable\n'
            'related: [kernel-gated-dual-gemm, technique-kernel-fusion, lang-triton-iluvatar]\n'
            f'sources: [{SOURCE}]\n'
            'aliases: ["BI-V150 SwiGLU", "Megatron BF16 gated activation"]\n')
    if not rounds:
        return text + 'performance_claims: []\n---\n\n'
    text += 'performance_claims:\n'
    for trial in rounds:
        for label, description in (
                ('vs_control', 'added SwiGLU versus same-round CE0008+gradnorm001'),
                ('vs_native', 'combined CE0008+gradnorm001+SwiGLU versus same-round native')):
            metric = ('Formal GPU-synchronized wall throughput ratio; candidate ' + trial['version']
                      + '; ' + description + '; round ' + trial['decision'])
            locator = (f"evidence/{BUNDLE}/megatron_bf16/evidence/swiglu-stage29/"
                       f"{trial['file']}#{label}")
            text += ('  - gpu: "Iluvatar BI-V150"\n    dtype: bf16\n'
                     '    shape: "4L H512 FFN2048 S128 batch2 V1024 SwiGLU; 17895936 parameters"\n'
                     f'    metric: {yaml_scalar(metric)}\n'
                     f"    value: {trial['ratios'][label]['ratio']!r}\n"
                     f'    source_id: {SOURCE}\n'
                     f'    source_locator: {yaml_scalar(locator)}\n')
    return text + '---\n\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill', type=Path, required=True)
    args = parser.parse_args()
    skill = args.skill.resolve()
    decision_path = RAW / 'decision.json'
    decision = read(decision_path)
    for name in ('status', 'promoted', 'candidate_versions', 'scope'):
        assert name in decision, 'sealed decision missing field: ' + name
    contract = read(SCRIPT_ROOT / 'contracts/main.json')
    assert contract['model']['parameters'] == 17895936
    assert contract['model']['gated_linear_unit'] is True
    assert contract['model']['bias_activation_fusion'] is True
    assert '--swiglu' in contract['training_argv']
    assert '--no-bias-swiglu-fusion' not in contract['training_argv']
    bundle = skill / 'evidence' / BUNDLE
    source_page = skill / 'sources/experiments' / (BUNDLE + '.md')
    kernel_page = skill / 'wiki/kernels/swiglu-megatron-bi-v150.md'
    assert not any(path.exists() for path in (bundle, source_page, kernel_page)), 'immutable delivery already exists'
    assert (skill / 'SKILL.md').is_file()
    assert (skill / 'references/iluvatar-migration-ledger.md').is_file()
    rounds = [summary for path in sorted(RAW.glob('performance-main-*.json'))
              if (summary := summarize_performance(path)) is not None]
    numerical = numerical_text()
    failures = failure_text()
    files, omitted = collect_files()
    bundle.mkdir(exist_ok=False)
    write_new(bundle / '.gitattributes', '* -text -eol -whitespace\n')
    records = []
    for path, relative in files:
        assert path.is_file(), 'missing experiment input: ' + str(path)
        dest = bundle / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        records.append({'path': relative, 'sha256': sha(dest), 'bytes': dest.stat().st_size})
    manifest = {'source_id': SOURCE, 'decision_sha256': sha(decision_path), 'files': records,
                'excluded_raw_files': omitted,
                'raw_policy': 'Tensor snapshots, activation .pt files, archives and compiled binaries '
                              'are excluded. Their inventories and hash backup receipts remain included.'}
    write_new(bundle / 'manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    date = decision.get('date', '2026-10-04')
    shared = '''The baseline is the pinned original Megatron default `bias_swiglu_impl`
with native `torch.compile` fusion. Fusion is not disabled to manufacture a
slower reference. Both GEMMs, mock-data/schedule/DDP/SGD and the native local
PyTorch route remain unchanged; the candidate replaces only the activation.

This is a new SwiGLU model: 4 layers, H512, FFN2048, S128, V1024, batch2,
**17,895,936 parameters**. Earlier GELU models had 13,701,632 parameters;
their absolute times and percentages are not a paired baseline for this model.
All arms in each reported round use the same new contract and initial seed.

The tested path is single BI-V150, CoreX4.2/vendor Torch2.4.1/Triton2.1,
TP/PP/CP1, true BF16 module weights, FP32 gradients and SGD master weights,
no autocast, dropout, clipping, bias, GLU offset, clamp or offloading. Native
compiled SwiGLU opmath is characterized separately from eager BF16 math;
do not infer its internal rounding from the earlier RMSNorm implementation.
'''
    body = ('# Native default SwiGLU, measured before replacement\n\n' + shared
            + '\n## Sealed decision\n\n'
            + f"Status: `{decision['status']}`; promoted: `{decision['promoted']}`.\n\n"
            + 'Candidate versions: `' + json.dumps(decision['candidate_versions'], ensure_ascii=False) + '`.\n\n'
            + 'Decision scope: ' + str(decision['scope']) + '\n\n'
            + '## Numerical and formal update gates\n\n'
            + 'Output, input gradients, model weights and FP32 masters are compared using '
              'CPU float64 copies at unchanged atol3e-4/rtol1e-3. The fixed independent '
              'probes also check input preservation, guards and fresh inputs; v3 has '
              '37 cases including strided gradients, offsets, alternating sizes and current streams. '
              'Case counts differ from individual output/gradient gate counts.\n\n'
            + numerical + '\n## Formal training throughput by round\n\n'
            + 'Each validated round has three fresh-process seeds and three rotated arms. '
              'Each process runs120 iterations; the first20 are discarded, and the ten complete '
              'interval10 means ending30..120 are retained. The original GPU-synchronized '
              '`time.time` interval timer is unchanged. No tensor snapshot hooks, profiler, '
              'evaluation or checkpoint run in timing. Candidate-side deployment dispatch '
              'and counters are included; the native/control paths are not artificially '
              'instrumented to make them slower. Three observations are not confidence intervals.\n\n'
            + performance_text(rounds)
            + '\n## Compiler, controller and numerical failures\n\n' + failures
            + '\n## Evidence and migration boundaries\n\n'
            + 'Claude Code CLI generated candidates using the project skill and the configured '
              'Paratera model; independent controllers executed and judged them. All original '
              'prompts, responses, rejected code, flags, logs, source hashes and verified backup '
              'receipts are retained. A compiler or lazy-loader failure is distinct from a '
              'numerical rejection and from a measured performance regression.\n\n'
            + 'Standalone paired `torch.cuda.Event` API elapsed measurements include possible '
              'host/autograd submission gaps. They are not isolated device-kernel latency or '
              'the formal training wall metric. Do not mix these interfaces or historical '
              'trial percentages. Promotion requires full update correctness, >1% additional '
              'throughput versus the same-round optimized control and every seed faster.\n\n'
            + 'The portable hypothesis is gate activation/elementwise fusion and reduced launch '
              'dispatch. Compiled-code launch caches must use current streams and fresh pointers; '
              'input/output/result caches are forbidden. C++ extensions are bound to actual source, '
              'controller build and library hashes. SDK CUDA labels and rewritten compiler flags '
              'do not prove a NVIDIA hardware identity. No NVFP4, TMEM, TMA, tcgen05, WGMMA or '
              'H100/GB200 timing is relabelled as BI-V150 evidence. No production, long-run '
              'convergence, AdamW, multi-GPU or unrestricted replacement claim follows.\n\n'
            + f'All small evidence is in `evidence/{BUNDLE}`. Raw tensors and binary artifacts '
              'remain outside Git; inventories and hash backup receipts are included.\n')
    write_new(source_page, source_front(skill, bundle, date) + body)
    kernel_body = ('# Preserve the native fused activation baseline\n\n' + shared
                   + '\nThe operation is `SiLU(gate) * up` after the existing FC1 output is '
                     'split into halves. This experiment does not fuse either GEMM, change '
                     'the MLP dimensions or insert NVFP4 quantization. Read the '
                     '[stage29 source](../../sources/experiments/bi-v150-corex42-stage29.md) '
                     'for every measured and rejected candidate.\n\n'
                   + '## Outcomes\n\n' + performance_text(rounds)
                   + '\nKeep native default fusion whenever a candidate has not passed both '
                     'the fixed training update gates and the per-round promotion rule. '
                     'A combined improvement versus original native can come from the existing '
                     'CE and gradient-norm optimizations while the new activation regresses.\n\n'
                   + '## Runnable native reference\n\n'
                   + 'With pinned Megatron on PYTHONPATH in the recorded CoreX environment:\n\n'
                   + '```python\nimport torch\n'
                     'from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl\n\n'
                     'x = torch.randn(128, 2, 4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)\n'
                     'y = bias_swiglu_impl(x, None, False, False)\n'
                     'dx = torch.autograd.grad(y, x, torch.randn_like(y))[0]\n'
                     'assert y.shape == (128, 2, 2048) and dx.shape == x.shape\n```\n'
                   + '\nIndependent controller probes, model-generated code and formal '
                     'wrappers are retained with source hashes. Rebuild native extensions '
                     'against the matching vendor ABI before a numerical probe; importing '
                     'a binary alone is not a correctness or performance result.\n')
    write_new(kernel_page, kernel_front(rounds) + kernel_body)
    addition = ('\nFor BI-V150 SwiGLU, retain the default compiled native baseline and distinguish '
                'the new17.9M gated model from earlier13.7M GELU trials; read '
                '[stage29](sources/experiments/bi-v150-corex42-stage29.md) and '
                '[the activation page](wiki/kernels/swiglu-megatron-bi-v150.md). '
                'Compiler/controller failures, numerical rejection and training regressions '
                'remain separate evidence; never infer throughput for an unmeasured candidate.\n')
    with (skill / 'SKILL.md').open('ab') as stream:
        stream.write(addition.encode('utf-8'))
    addition = ('\n### Stage29: native default fused SwiGLU\n\n'
                'Preserve pinned default torch.compile fusion and both native GEMMs. '
                'SwiGLU changes the model to17,895,936parameters; compare only same-contract '
                'arms. Bind actual candidate/extension/build hashes; validate fresh pointers, '
                'BF16 values/gradients and full SGD/master updates before synchronized '
                'training throughput. Portable fusion and code-launch caching do not establish '
                'NVFP4/TMEM/tcgen05 equivalence. [Evidence](../sources/experiments/' + BUNDLE + '.md).\n')
    with (skill / 'references/iluvatar-migration-ledger.md').open('ab') as stream:
        stream.write(addition.encode('utf-8'))
    print(json.dumps({'source_id': SOURCE, 'kernel_id': KERNEL_ID,
                      'files': len(records), 'bundle_bytes': sum(row['bytes'] for row in records),
                      'formal_rounds': len(rounds), 'decision_status': decision['status'],
                      'promoted': decision['promoted']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
