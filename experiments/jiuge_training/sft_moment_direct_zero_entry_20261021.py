"""Use the CPU-tested direct-zero candidate in a fresh guarded35-step SFT run."""
import json
from pathlib import Path
from sft_moment_entry_20261021 import derive as prior_entry

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source,expected):
    text=prior_entry(source,expected)
    changes=[
        ('from sft_local_moments_20261021 import install','from sft_local_moments_v3_20261021 import install'),
        ('    moment_events = []',
         '    import sft_local_moments_v3_20261021 as moment_module\n'
         '    from sft_moment_memory_guard_20261021 import MeteredMomentEvents,snapshot as memory_snapshot,install as install_memory_guard\n'
         '    moment_events = MeteredMomentEvents(root,rank)\n'
         "    undo_memory_guard=install_memory_guard(moment_module) if args.sft_moment_arm=='local-moments' else None"),
        ("        value = observe(opt, phase, torch)","        value = observe(opt, phase, torch)\n        value['memory_budget_snapshot']=memory_snapshot()"),
        ('        result = old_step(opt, *values, **keywords)',
         "        if steps==0:\n            with (root/f'memory-before-first-update-rank-{rank}.json').open('x') as stream:json.dump(memory_snapshot(),stream,indent=2)\n"
         '        result = old_step(opt, *values, **keywords)'),
        ('        undo_moments()','        undo_moments()\n    if undo_memory_guard is not None:\n        undo_memory_guard()'),
    ]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact direct-zero entry derivation required: '+before)
        text=text.replace(before,after)
    return text


def main():
    files=json.loads((PREP/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']
    text=derive((PREP/'sft_numa_entry_20261021.py').read_bytes(),files['sft_numa_entry_20261021.py'])
    exec(compile(text,'<guarded-direct-zero-SFT-entry>','exec'),{'__name__':'__main__','__file__':__file__})


if __name__=='__main__':main()
