"""Independent slow-rank timer/CV/loss recomputation from bounded formal keys."""
from pathlib import Path

BASE = Path(__file__).resolve().parent


def generated():
    source = (BASE/'review_sft_phased_acceptance_20261024.py').read_text(encoding='utf-8')
    begin = source.index("    primary = read(root/'numeric/primary-archive-verification.json')")
    end = source.index('    rounds = []', begin)
    proof = '''    evidence = root/'correctness-evidence'
    closed = read(evidence/'closure.json')
    independent = read(evidence/'independent-review.json')
    admission = read(root/'formal-admission.json')
    primary = read(evidence/'primary-archive-verification.json')
    semantic = read(evidence/'semantics-verification.json')
    assert closed['passed'] and independent['passed']
    assert independent['full_state_and_native_resume_semantics_passed']
    assert len(independent['numeric_runs'])==10
    assert all(r['max_abs_error']==0 for r in independent['numeric_runs'])
    assert primary['passed'] and primary['unique_raw_chunks_verified']==18601
    assert primary['primary_complete_raw_values_verified'] and primary['checkpoint_resume_tested']
    assert semantic['passed'] and semantic['full_model_native_checkpoint_resume_verified']
    assert admission['correctness_key_bundle_sha256']==closed['key_bundle_sha256']==independent['key_bundle_sha256']
    assert admission['correctness_independent_review_sha256']==hashlib.sha256((evidence/'independent-review.json').read_bytes()).hexdigest()
    old = read(BASE/'results/sft-phased-acceptance-source-manifest-20261024-02.json')['files']
    assert len(old)==169 and admission['correctness_original_sources_sha256']==old
    assert all(frozen[n]==h for n,h in old.items())
    assert controller['correctness_complete_before_formal'] and controller['new_correctness_GPU_runs']==0
    assert not controller['old_performance_reused']
    for folder in (DEST/'runs').iterdir():
        assert not list(folder.glob('audit-cache-rank-*.json'))
'''
    source = source[:begin]+proof+source[end:]
    changes = {
        'sft-phased-acceptance-driver-20261024-01':'sft-phased-formal-driver-20261025-01',
        'sft-phased-acceptance-verified-20261024-01':'sft-phased-formal-verified-20261025-01',
        'sft-phased-acceptance-20261024-01':'sft-phased-formal-20261025-01',
        'sft-phased-acceptance-source-manifest-20261024-01.json':'sft-phased-formal-source-manifest-20261025-01.json',
        "'phased20261024'+name":"'phased20261025'+name",
        'sft-phased-acceptance-independent-review-20261024-01.json':'sft-phased-formal-independent-review-20261025-01.json',
        'fresh_SFT_state_semantics_and_native_resume_passed':'own_closed_SFT_state_semantics_and_native_resume_passed',
        'from pinning_key_results import verify':"from pinning_key_results import verify, KEY_SUFFIXES\nKEY_SUFFIXES.add('.sh')",
    }
    for before, after in changes.items():
        assert source.count(before)==1, before
        source = source.replace(before, after)
    return source


if __name__ == '__main__':
    exec(compile(generated(), '<independent-formal-only-review>', 'exec'),
         {'__name__': '__main__', '__file__': __file__})
