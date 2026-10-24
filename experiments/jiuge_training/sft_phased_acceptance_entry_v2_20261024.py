"""Untimed cache-safe audit; formal entry remains the screened V4 implementation."""
from pathlib import Path
import os
import sys
from sft_phased_acceptance_recovery_20261024 import configure


def main():
    core = configure()
    import sft_phased_acceptance_entry_20261024 as previous
    stage = sys.argv[sys.argv.index('--sft-pinning-mode')+1]
    if stage=='formal':
        previous.main()
        return
    if stage not in ('audit','resume'):
        raise ValueError('Cache observer may only run in untimed audit/resume')
    run_root=Path(sys.argv[sys.argv.index('--jiuge-run-root')+1])
    from sft_phased_audit_cache_20261024 import OwnAuditCache
    cache = OwnAuditCache(core.ROOT/'numeric',core.ROOT/'checkpoints')
    undo = cache.install()
    original = previous.install_steps
    def install(*args, **kwargs):
        state,finish = original(*args,**kwargs)
        hdo = args[1]
        stepped = hdo.step
        first = True
        def guarded(opt,*values,**keywords):
            nonlocal first
            if first:
                cache.first_update()
                first=False
            return stepped(opt,*values,**keywords)
        hdo.step=guarded
        return state,finish
    previous.install_steps=install
    try:
        previous.main()
    finally:
        previous.install_steps=original
        undo()
        cache.save(run_root/f'audit-cache-rank-{os.environ["RANK"]}.json')


if __name__=='__main__':
    main()
