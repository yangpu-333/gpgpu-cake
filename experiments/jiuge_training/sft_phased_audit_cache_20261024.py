"""Reclaim only clean cache of this campaign's own untimed numeric originals.

No global drop_caches, page migration, tensor mutation or lowered memory guard.
This instrumentation is forbidden in formal timing.
"""
import json
import os
from pathlib import Path


class OwnAuditCache:
    def __init__(self, numeric, checkpoint):
        self.numeric = Path(numeric).resolve(strict=True)
        self.checkpoint = Path(checkpoint).resolve()
        self.calls = 0
        self.files = set()
        self.before_update = []

    def memory(self):
        root = Path('/sys/fs/cgroup')
        return {n:(root/n).read_text() for n in ('memory.current','memory.max','memory.stat','memory.events')}

    def advise(self, path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError('Symlinks are not eligible')
        path = path.resolve(strict=True)
        if (not path.is_file()
                or not (path.is_relative_to(self.numeric) and path.suffix=='.zlibpack'
                        or path.is_relative_to(self.checkpoint))):
            raise ValueError('Only this campaign numeric packs/checkpoints may receive advice')
        with path.open('rb') as stream:
            os.posix_fadvise(stream.fileno(),0,0,os.POSIX_FADV_DONTNEED)
        self.calls += 1
        self.files.add(str(path))

    def install(self):
        import pinning_full_audit
        previous = pinning_full_audit.FullAudit.tensor
        owner = self
        def tensor(audit, field, value):
            row = previous(audit,field,value)
            reference = audit.reference.get(field)
            if reference:
                for path in {audit.campaign/chunk['file'] for chunk in reference['chunks']}:
                    owner.advise(path)
            return row
        pinning_full_audit.FullAudit.tensor = tensor
        def undo():
            if pinning_full_audit.FullAudit.tensor is not tensor:
                raise ValueError('Unexpected audit observer replacement')
            pinning_full_audit.FullAudit.tensor = previous
        return undo

    def first_update(self):
        before = self.memory()
        for path in self.numeric.glob('native/rank-*/raw.zlibpack'):
            self.advise(path)
        if self.checkpoint.is_dir():
            for path in self.checkpoint.rglob('*'):
                if path.is_file():
                    self.advise(path)
        self.before_update.append({'before':before,'after':self.memory(),
                                   'advice_is_optional_not_a_memory_reservation':True})

    def save(self, path):
        result = {'schema':'jiuge7b-owned-audit-cache-advice-v1','advice_calls':self.calls,
                  'owned_files':sorted(self.files),'first_update_memory':self.before_update,
                  'payload_modified':False,'global_memory_policy_modified':False,
                  'production_8GiB_and_16GiB_guards_unchanged':True,'installed_in_formal_timing':False}
        with Path(path).open('x') as stream:
            json.dump(result,stream,indent=2)
