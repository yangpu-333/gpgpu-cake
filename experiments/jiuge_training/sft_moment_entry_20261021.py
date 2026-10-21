"""Add experimental moment storage to the unchanged NUMA capture entry."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source,expected):
    if hashlib.sha256(source).hexdigest()!=expected:
        raise ValueError('Frozen NUMA entry differs')
    changes = [
        ("    args, remaining = parser.parse_known_args()", "    parser.add_argument('--sft-moment-arm',choices=('baseline','affinity-control','local-moments'),required=True)\n    args, remaining = parser.parse_known_args()\n    if args.sft_numa_mode != ('unbound' if args.sft_moment_arm=='baseline' else 'gpu-local8'):\n        raise ValueError('Frozen arm-to-affinity mapping required')"),
        ('    import torch\n', "    import torch\n    from sft_local_moments_20261021 import install\n    moment_events = []\n    undo_moments = install(torch,moment_events) if args.sft_moment_arm=='local-moments' else None\n"),
        ("    with (root/f'sft-numa-observer-rank-{rank}.json').open('x') as stream:", "    if undo_moments is not None:\n        undo_moments()\n    with (root/f'sft-local-moments-rank-{rank}.json').open('x') as stream:\n        json.dump({'arm':args.sft_moment_arm,'storage_only_candidate':True,'moment_events':moment_events,'formal_performance_evidence':False},stream,indent=2)\n    with (root/f'sft-numa-observer-rank-{rank}.json').open('x') as stream:")]
    text = source.decode()
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact moment entry derivation failed: '+before)
        text = text.replace(before,after)
    return text


def main():
    import json
    files=json.loads((PREP/'sft-moment-screen-source-manifest-20261021.json').read_text())['files']
    text=derive((PREP/'sft_numa_entry_20261021.py').read_bytes(),files['sft_numa_entry_20261021.py'])
    exec(compile(text,str(PREP/'sft_moment_entry_20261021.generated.py'),'exec'),
         {'__name__':'__main__','__file__':__file__})


if __name__=='__main__':main()
