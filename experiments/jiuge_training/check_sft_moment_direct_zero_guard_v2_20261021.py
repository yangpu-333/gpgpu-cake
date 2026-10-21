"""CPU16GiB fixture uses512MiB reserve; SFT production reserve stays8GiB.

The production8GiB guard correctly refused the first CPU fixture because file
cache left less than8GiB cgroup headroom. Preserve that refusal as evidence.
"""
from check_sft_moment_direct_zero_guard_20261021 import generated as previous


def generated():
    text=previous()
    changes=[
        ('sft-moment-direct-zero-preparation-20261021-01/guarded-regression',
         'sft-moment-direct-zero-preparation-20261021-01/guarded-regression-v2'),
        ('    undo_guard=install_guard(guarded_module)',
         '    import sft_moment_memory_guard_20261021 as guard_module\n'
         '    assert guard_module.RESERVE_BYTES==8*1024**3\n'
         '    guard_module.RESERVE_BYTES=512*1024**2  # CPU fixture only, never the SFT entry.\n'
         '    undo_guard=install_guard(guarded_module)'),
        ('        undo_guard()', '        undo_guard()\n        guard_module.RESERVE_BYTES=8*1024**3'),
        ("    value = {'passed':True", "    value = {'CPU_fixture_reserve_bytes':512*1024**2,'SFT_production_reserve_bytes':8*1024**3,'passed':True"),
    ]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact scaled CPU guard fixture required: '+before)
        text=text.replace(before,after)
    return text


if __name__=='__main__':
    exec(compile(generated(),'<scaled-CPU-only-live-allocation-guard-fixture>','exec'),
         {'__name__':'__main__','__file__':__file__})
