"""Reuse the exact CPU equality/resume fixtures with the live memory guard."""
from check_sft_local_moments_v3_20261021 import generated as previous


def generated():
    text=previous()
    changes=[
        ('sft-moment-direct-zero-preparation-20261021-01/cpu-regression',
         'sft-moment-direct-zero-preparation-20261021-01/guarded-regression'),
        ('    uninstall = install(torch,events)',
         '    import sft_local_moments_v3_20261021 as guarded_module\n'
         '    from sft_moment_memory_guard_20261021 import install as install_guard\n'
         '    undo_guard=install_guard(guarded_module)\n    uninstall = install(torch,events)'),
        ('        uninstall()', '        uninstall()\n        undo_guard()'),
    ]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact guarded CPU fixture required: '+before)
        text=text.replace(before,after)
    return text


if __name__=='__main__':
    exec(compile(generated(),'<CPU-state-and-resume-with-live-allocation-guard>','exec'),
         {'__name__':'__main__','__file__':__file__})
