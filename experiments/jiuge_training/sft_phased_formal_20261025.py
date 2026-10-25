"""Formal-only continuation of the closed Oct24 numerical acceptance."""
import hashlib
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-phased-formal-20261025-01'
APPROVAL = PREP/'sft-phased-formal-approved-20261025-01.json'
MANIFEST = 'sft-phased-formal-source-manifest-20261025-01.json'
EVIDENCE = PREP/'sft-phased-formal-evidence-20261025-01'
OLD = PREP/'sft-phased-acceptance-20261024-02'
CPU = PREP/'sft-phased-correctness-postprocess-20261024-03'
KEY_SHA = '26dada4e9e162803b23b5be0260007444d0da5bdb2dd31b10439a3ed2d84f272'
PAIR = ['GPU-c9db5c06-61c6-01b7-b9aa-20307600482c',
        'GPU-4ce9cbee-8444-67c3-2ead-dfda2618ccdf']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def configure():
    import sft_phased_acceptance_20261024 as core
    core.ROOT, core.APPROVAL, core.MANIFEST = ROOT, APPROVAL, MANIFEST
    return core


def evidence_gate():
    old_sources = read(PREP/'sft-phased-acceptance-source-manifest-20261024-02.json')['files']
    assert len(old_sources) == 169
    for name, digest in old_sources.items():
        assert sha(PREP/name) == sha(OLD/'source'/name) == digest, name
    cpu_sources = read(EVIDENCE/'cpu-source-manifest.json')['files']
    for name, digest in cpu_sources.items():
        assert sha(PREP/name) == digest, name
    closure = read(EVIDENCE/'closure.json')
    independent = read(EVIDENCE/'independent-review.json')
    local = read(EVIDENCE/'local-key-verification.json')
    assert closure['passed'] and closure['full_state_and_native_resume_semantics_passed']
    assert closure['all10_numeric_runs_max_absolute_error'] == 0
    assert independent['passed'] and independent['full_state_and_native_resume_semantics_passed']
    assert len(independent['numeric_runs']) == 10 and independent['ACT_raw_chunks_verified'] == 18601
    assert all(row['max_abs_error'] == 0 for row in independent['numeric_runs'])
    assert local['passed'] and not local['full_raw_local_copy_verified']
    assert local['bundle_sha256'] == independent['key_bundle_sha256'] == closure['key_bundle_sha256'] == KEY_SHA
    assert sha(CPU/'correctness-key-results.zip') == KEY_SHA
    assert read(CPU/'controller.json')['status'] == 'completed'
    primary = read(CPU/'primary-archive-verification.json')
    semantic = read(CPU/'semantics-verification.json')
    assert primary['passed'] and primary['unique_raw_chunks_verified'] == 18601
    assert primary['primary_complete_raw_values_verified'] and primary['checkpoint_resume_tested']
    assert len(primary['runs']) == 10
    assert semantic['passed'] and semantic['full_model_native_checkpoint_resume_verified']
    for row in independent['numeric_runs']:
        path = OLD/'numeric'/row['arm']/f"rank-{row['rank']}"/'all-state-manifest.json'
        assert sha(path) == row['manifest_sha256']
    checkpoint = read(CPU/'checkpoint-sha256.json')
    assert len(checkpoint['files']) == 5 and sum(row['bytes'] for row in checkpoint['files']) == 107368611274
    for row in checkpoint['files']:
        assert (Path(checkpoint['checkpoint'])/row['file']).stat().st_size == row['bytes']
    from admit_pinning_formal import admit
    result = admit(CPU/'primary-archive-verification.json', CPU/'semantics-verification.json',
                   CPU/'correctness-key-results.zip', EVIDENCE/'local-key-verification.json')
    result.update(objective='answer_only_SFT',
                  sft_data_receipt_sha256=sha(PREP/'datasets/alpaca-sft-answer-v1-receipt.json'),
                  incremental_phased_candidate_accepted=True,
                  incremental_source_sha256=configure().verify_sources(),
                  incremental_baseline='accepted pinned, unbound',
                  correctness_original_sources_sha256=old_sources,
                  correctness_key_bundle_sha256=KEY_SHA,
                  correctness_independent_review_sha256=sha(EVIDENCE/'independent-review.json'),
                  reuse='closed Oct24 full state/native resume/raw SHA; no old performance reused')
    return result
