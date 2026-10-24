"""Independent V2 postprocessing: preserve original acceptance and reference provenance."""
from pathlib import Path
import json

BASE = Path(__file__).resolve().parent


def reference_gate():
    root = BASE/'results/sft-phased-acceptance-verified-20261024-02/sft-phased-acceptance-20261024-02'
    origins = json.loads((root/'reference-origins.json').read_text(encoding='utf-8'))
    assert origins['same_campaign_references_not_new_runs']
    assert origins['old158_sources_unchanged'] and origins['candidate_failed_attempt_not_reused']
    assert not origins['performance_results_reused'] and origins['files']
    assert all(row['same_inode'] for row in origins['files'])
    old = json.loads((BASE/'results/sft-phased-acceptance-source-manifest-20261024-01.json').read_text(encoding='utf-8'))['files']
    new = json.loads((BASE/'results/sft-phased-acceptance-source-manifest-20261024-02.json').read_text(encoding='utf-8'))['files']
    assert len(old)==158 and len(new)==169 and all(new[n]==h for n,h in old.items())
    runs = root.parent/'runs'
    formal_folders = list(runs.glob('pinning-formal-phased20261024v2*-*'))
    assert len(formal_folders) in (9,18)
    for folder in formal_folders:
        assert not list(folder.glob('audit-cache-rank-*.json')), 'No audit cache advice in formal timing'
    for rank in (0,1):
        cache = json.loads((runs/'phased-audit-optimized-20261024-02'/f'audit-cache-rank-{rank}.json').read_text(encoding='utf-8'))
        assert cache['production_8GiB_and_16GiB_guards_unchanged'] and cache['first_update_memory']
        assert not cache['payload_modified'] and not cache['global_memory_policy_modified']
        assert not cache['installed_in_formal_timing']


def main():
    text = (BASE/'review_sft_phased_acceptance_20261024.py').read_text(encoding='utf-8')
    changes = {
        'sft-phased-acceptance-driver-20261024-01':'sft-phased-acceptance-driver-20261024-02',
        'sft-phased-acceptance-verified-20261024-01':'sft-phased-acceptance-verified-20261024-02',
        'sft-phased-acceptance-20261024-01':'sft-phased-acceptance-20261024-02',
        'sft-phased-acceptance-source-manifest-20261024-01.json':'sft-phased-acceptance-source-manifest-20261024-02.json',
        "'phased20261024'+name":"'phased20261024v2'+name",
        'sft-phased-acceptance-independent-review-20261024-01.json':'sft-phased-acceptance-independent-review-20261024-02.json',
        "    result = {'schema':":"    reference_gate()\n    result = {'schema':",
    }
    for old,new in changes.items():
        assert text.count(old)==1, old
        text = text.replace(old,new)
    exec(compile(text,'<independent-acceptance-review-v2>','exec'),
         {'__name__':'__main__','__file__':__file__,'reference_gate':reference_gate})


if __name__=='__main__':
    main()
