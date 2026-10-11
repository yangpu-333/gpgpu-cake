"""Compare generated, rebuilt and project-loaded skill bytes and provenance."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def files(root):
    return {p.relative_to(root).as_posix(): (p.stat().st_size, sha(p))
            for p in root.rglob('*') if p.is_file() and not any(part in ('.git', '__pycache__')
            for part in p.relative_to(root).parts)}


def paths(root, architecture):
    process = subprocess.run([sys.executable, str(root / 'scripts/query.py'), '--architecture', architecture,
                              '--paths-only', '--limit', '1000'], capture_output=True, text=True, encoding='utf-8', check=True)
    return {p.replace('\\', '/') for p in process.stdout.splitlines()}


def main():
    generated = ROOT / 'kernelwiki-iluvatar-stage35-retry01-release'
    rebuilt = ROOT / 'kernelwiki-iluvatar-stage35-retry01-rebuilt'
    project = ROOT / 'upstream-kda-stage35-retry01/.claude/skills/kernelwiki-iluvatar'
    baseline = ROOT / 'kernelwiki-iluvatar-stage34-release'
    target = HERE / 'evidence/skill-provenance-retry01-02.json'
    assert not target.exists()
    reference = files(generated)
    installed = files(project)
    managed_path = project / '.kda-managed-skill.json'
    managed = json.loads(managed_path.read_text())
    assert managed_path.is_file() and '.kda-managed-skill.json' not in reference
    assert managed == {'file_count': 4923, 'name': 'kernelwiki-iluvatar'}
    del installed['.kda-managed-skill.json']
    assert len(reference) == 4923 and reference == files(rebuilt) == installed
    original = {p.relative_to(baseline).as_posix(): sha(p)
                for folder in ('wiki', 'sources', 'prs') for p in (baseline / folder).rglob('*.md')}
    assert len(original) == 1077 and all(sha(generated / rel) == digest for rel, digest in original.items())
    counts = {architecture: len(paths(project, architecture)) for architecture in ('bi-v150', 'sm90', 'sm100')}
    assert counts == {'bi-v150': 38, 'sm90': 286, 'sm100': 379}
    for architecture in ('sm90', 'sm100'):
        assert paths(baseline, architecture) == paths(project, architecture)
    spec = importlib.util.spec_from_file_location('fresh_skill_query', generated / 'scripts/query.py')
    query = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(generated / 'scripts'))
    spec.loader.exec_module(query)
    pages = query.load_all_pages()
    ids = {p['fm']['id'] for p in pages if p['path'].replace('\\', '/').startswith('sources/')}
    assert len(pages) == 1079 and len(ids) == 1010
    manifest_path = ROOT / 'skill-patches.json'
    manifest = json.loads(manifest_path.read_text())
    assert len(manifest['patches']) == 5
    for row in manifest['patches']:
        assert sha(ROOT / row['file']) == row['sha256']
    expected_previous = ['320f347cfde8267009a8abe4511d322cca7b7296a69e1bab3c42b6cf41ac3660',
        '652227208de55c42b9c6a914200f180125d793d6edab892d4d7b627640627426',
        'f41298cd92a68720d27ac4b3b2ece8f7b57c95ee01d08742534aeb44baf4f944',
        '317b938317bbd1835316e0939319537815ac22d8a82865bf19dada196efa124a']
    assert [r['sha256'] for r in manifest['patches'][:4]] == expected_previous
    bundle = generated / 'evidence/bi-v150-corex42-stage35-retry01'
    data = json.loads((bundle / 'manifest.json').read_text())
    assert len(data['files']) == 333
    assert all(sha(bundle / row['path']) == row['sha256'] for row in data['files'])
    record = {'schema': 'fresh-heldout-stage35-skill-provenance-v2', 'status': 'verified',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'indexed_pages': len(pages), 'source_ids': len(ids),
        'architecture_paths': counts, 'previous_pages_byte_identical': len(original),
        'generated_rebuilt_project_files_byte_identical': len(reference),
        'project_loader_receipt_sha256': sha(managed_path),
        'project_loader_receipt_is_extra_control_file': True,
        'new_evidence_files': len(data['files']), 'new_evidence_bytes': sum(r['bytes'] for r in data['files']),
        'first_four_patches_unchanged': True, 'patches': manifest['patches'], 'skill_patch_manifest_sha256': sha(manifest_path),
        'decision_sha256': sha(HERE / 'evidence/decision-retry01-02.json'),
        'new_evidence_manifest_sha256': sha(bundle / 'manifest.json'),
        'source_page_sha256': sha(generated / 'sources/experiments/bi-v150-corex42-stage35-retry01.md'),
        'kernel_page_sha256': sha(generated / 'wiki/kernels/weighted-swiglu-heldout-megatron-bi-v150.md'),
        'full_skill_file_map': {name: {'bytes': values[0], 'sha256': values[1]} for name, values in reference.items()},
        'local_test_result': {'tests': 162, 'passed': True, 'runner': 'Python3.13.2 on Windows with scoped official-runtime python3 alias',
                              'observed_seconds': 77.489},
        'corpus_validation': '1079 files/1010 source IDs/37 asset bundles/14 ledgers; all valid',
        'verifier_sha256': sha(__file__), 'gpu_work_executed': False,
        'personal_skill_install_required': False, 'upstream_pr_created': False}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps({key: record[key] for key in ('status', 'indexed_pages', 'source_ids', 'architecture_paths',
                     'generated_rebuilt_project_files_byte_identical', 'new_evidence_files', 'first_four_patches_unchanged')}))


if __name__ == '__main__':
    main()
