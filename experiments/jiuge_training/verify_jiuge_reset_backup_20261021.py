"""Offline hash and restoration inventory verification; no training or network."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import zipfile

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results/reset-backup-20261021'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    prior=json.loads((ROOT/'status.json').read_text())
    recovery=json.loads((BASE/'results/jiuge-reset-recovery-verification-20261021.json').read_text())
    rows=prior['verified']
    for row in rows:
        path=ROOT/Path(row['local']).name
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    archive=ROOT/Path(recovery['local_bundle']).name
    assert archive.stat().st_size==recovery['bytes'] and sha(archive)==recovery['sha256']
    sources=json.loads((BASE/'results/sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('recovery-manifest.json'))
        assert len(z.namelist())==len(set(z.namelist()))
        assert set(z.namelist())==set(manifest['files'])|{'recovery-manifest.json'}
        for name,expected in manifest['files'].items():
            relative=PurePosixPath(name)
            assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name
            with z.open(name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==expected['sha256'],name
        for name,expected in sources.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
            assert sha(BASE/name)==expected,name
    result={'schema':'jiuge7b-reset-local-offline-verification-v1',
            'verified_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
            'trace_delivery_bundles':rows,'recovery_archive':{k:v for k,v in recovery.items() if k!='metadata'},
            'archive_count':len(rows)+1,'verified_total_bytes':sum(r['bytes'] for r in rows)+recovery['bytes'],
            'recovery_member_sha_verified':len(manifest['files']),
            'frozen_candidate_sources_recoverable':len(sources),
            'full_training_checkpoints_locally_backed_up':False,
            'checkpoint_originals_location':'ACT shared Ceph;not container temporary directory',
            'GPU_used':False,'new_GPU_experiment_started':False,
            'new_7B_OOM_fix_validated':False,'formal_performance_evidence_for_new_candidate':False}
    target=BASE/'results/jiuge-reset-offline-verification-20261021.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trace_delivery_bundles','recovery_archive')}))


if __name__=='__main__':main()
