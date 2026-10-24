"""Identities and exact source derivations for a fresh incremental SFT audit."""
import hashlib
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-phased-acceptance-20261024-01'
APPROVAL = PREP/'sft-phased-acceptance-approved-20261024-01.json'
MANIFEST = 'sft-phased-acceptance-source-manifest-20261024-01.json'
POLICIES = {'native': 'baseline', 'control': 'affinity-control', 'optimized': 'local-moments'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024**2):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def replace_exact(text, changes):
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact frozen derivation required: '+before)
        text = text.replace(before, after)
    return text


def pinned_backend(payload, formal=False):
    """Keep audit arm identities; all incremental arms use accepted pinning."""
    text = payload.decode()
    if formal:
        changes = [
            ("if args.pinning_arm != 'native':", 'if True:'),
            ("name = 'candidate_hybrid_optimizer.py' if args.pinning_arm == 'optimized' else 'native_hybrid_optimizer.py'",
             "name = 'candidate_hybrid_optimizer.py'"),
            ("expected = '8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88' if args.pinning_arm == 'optimized' else EXPECTED",
             "expected = '8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88'"),
            ("if args.pinning_arm == 'optimized' and not all(p.is_pinned() for p in masters):",
             'if not all(p.is_pinned() for p in masters):'),
        ]
    else:
        changes = [
            ("if args.pinning_arm in ('control', 'optimized'):", 'if True:'),
            ("name = 'candidate_hybrid_optimizer.py' if args.pinning_arm == 'optimized' else 'native_hybrid_optimizer.py'",
             "name = 'candidate_hybrid_optimizer.py'"),
            ("expected = json.loads((args.pinning_campaign / 'candidate.json').read_text())['candidate_sha256'] if args.pinning_arm == 'optimized' else EXPECTED",
             "expected = json.loads((args.pinning_campaign / 'candidate.json').read_text())['candidate_sha256']"),
            ("if args.pinning_arm == 'optimized' and receipt['pinned_count'] != len(values):",
             "if receipt['pinned_count'] != len(values):"),
        ]
    return replace_exact(text, changes)


def verify_sources():
    manifest = read(PREP/MANIFEST)
    for name, digest in manifest['files'].items():
        if sha(PREP/name) != digest:
            raise ValueError('Frozen acceptance source changed: '+name)
    return manifest['files']


def evaluate_incremental(means):
    from analyze_pinning_formal import evaluate
    result = evaluate(means)
    result['round_promotion_passed'] = all(
        result[key]['all_seeds_faster'] and result[key]['geomean_throughput_ratio'] > 1.01
        for key in ('vs_native', 'vs_control'))
    result['native_role_means'] = 'accepted pinned, unbound incremental baseline'
    result['control_role_means'] = 'same accepted pinned, two-phase affinity'
    return result
