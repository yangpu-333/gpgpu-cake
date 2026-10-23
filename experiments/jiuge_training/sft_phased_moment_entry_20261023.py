"""Initialize on the opposite socket, then place CPU updates and moments locally."""
import json
from pathlib import Path
import sys
from sft_moment_direct_zero_entry_20261021 import derive as previous

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source, expected, updates):
    if updates not in (1,35):
        raise ValueError('Only standalone admission or unchanged35-step screen')
    text = previous(source,expected)
    changes = (
        ('from sft_numa_placement_20261021 import bind',
         'from sft_phased_placement_20261023 import prepare as bind,activate,install_physical_guard'),
        ("root/f'sft-numa-binding-rank-{rank}.json'","root/f'sft-numa-setup-binding-rank-{rank}.json'"),
        ("    undo_moments = install(torch,moment_events) if args.sft_moment_arm=='local-moments' else None",
         '    undo_moments = None\n    undo_physical_guard = None'),
        ('        nonlocal active, steps','        nonlocal active, steps, receipt, undo_moments, undo_physical_guard'),
        ('        if steps==0:\n',
         "        if steps==0:\n            receipt=activate(args.sft_numa_plan,args.sft_numa_mode,rank,root)\n            if args.sft_moment_arm=='local-moments':\n                undo_physical_guard=install_physical_guard(moment_module,receipt['selected_cpu_nodes'][0])\n                undo_moments=install(torch,moment_events)\n"),
        ('    if undo_memory_guard is not None:',
         '    if undo_physical_guard is not None:\n        undo_physical_guard()\n    if undo_memory_guard is not None:'),
    )
    for before,after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact phased entry required: '+before)
        text = text.replace(before,after)
    if updates == 1:
        for before,after in (
            ("if '--capture-windows' not in remaining or '--pinning-arm' not in remaining:",
             "if '--capture-windows' in remaining or '--pinning-arm' not in remaining:"),
            ('optimizer is None or steps != 35','optimizer is None or steps != 1'),
            ('All35 real updates and a live optimizer required','Exactly one real update and a live optimizer required')):
            if text.count(before) != 1:
                raise ValueError('Exact standalone update derivation required')
            text=text.replace(before,after)
    return text


if __name__ == '__main__':
    from sft_memory_snapshot_compat_20261023 import install
    install()
    updates=int(sys.argv[sys.argv.index('--train-iters')+1])
    files=json.loads((PREP/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']
    text=derive((PREP/'sft_numa_entry_20261021.py').read_bytes(),files['sft_numa_entry_20261021.py'],updates)
    exec(compile(text,'<phased-setup-local-moments>','exec'),{'__name__':'__main__','__file__':__file__})
