"""CPU-only: prove archive bytes/values persist and foreign cache paths are rejected."""
import hashlib
import json
from pathlib import Path

from sft_phased_acceptance_20261024 import PREP


def main():
    import torch
    from pinning_full_audit import FullAudit
    from sft_phased_audit_cache_20261024 import OwnAuditCache
    from sft_moment_memory_guard_20261021 import check_values,RESERVE_BYTES
    from sft_phased_placement_20261023 import NODE_RESERVE_BYTES
    assert not torch.cuda.is_initialized()
    assert RESERVE_BYTES==8*1024**3 and NODE_RESERVE_BYTES==16*1024**3
    root=PREP/'sft-audit-cache-CPU-20261024-01'
    root.mkdir(exist_ok=False)
    numeric=root/'numeric';numeric.mkdir()
    value=torch.arange(262144,dtype=torch.float32)
    baseline=FullAudit(numeric,'native',0)
    baseline.tensor('probe',value);baseline.flush();baseline.close()
    raw=numeric/'native/rank-0/raw.zlibpack'
    before=hashlib.sha256(raw.read_bytes()).hexdigest()
    cache=OwnAuditCache(numeric,root/'checkpoints')
    previous=FullAudit.tensor
    undo=cache.install()
    candidate=FullAudit(numeric,'optimized',0)
    try:
        row=candidate.tensor('probe',value.clone())
        assert row['passed'] and row['max_abs']==0
        candidate.flush()
        cache.first_update()
        external=root/'not-an-eligible-archive.txt';external.write_text('own test sentinel')
        try:
            cache.advise(external)
        except ValueError:
            rejected=True
        else:
            rejected=False
        assert rejected
    finally:
        candidate.close();undo()
    assert FullAudit.tensor is previous
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==before
    try:
        check_values(300*1024**3-2931544064,300*1024**3,526475264)
    except MemoryError:
        guard_rejected=True
    else:
        guard_rejected=False
    assert guard_rejected and not torch.cuda.is_initialized()
    cache.save(root/'advice.json')
    result={'passed':True,'GPU_used':False,'toy_numeric_max_abs':0,'archive_SHA_unchanged':True,
            'foreign_path_rejected':True,'original8GiB_guard_still_rejects_recorded_low_headroom':True,
            'old_full_audit_restored':True,'advice_calls':cache.calls,'full7B_acceptance':False}
    with (root/'summary.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
