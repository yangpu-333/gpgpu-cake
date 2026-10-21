"""Small ACT recovery snapshot; exclude weights, venv, credentials and big checkpoints."""
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import zipfile

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'reset-recovery-20261021-01'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    ROOT.mkdir(exist_ok=False)
    selected={}
    for folder in [PREP/'datasets',PREP/'megatron-native-5be9626',
                   PREP/'sft-moment-direct-zero-preparation-20261021-01',
                   PREP/'sft-moment-placement-preparation-20261021-01']:
        for path in folder.rglob('*'):
            if path.is_symlink():continue
            if path.is_file() and not any(x in path.parts for x in ('.git','__pycache__','.venv')):
                if path.stat().st_size>64*1024**2:raise ValueError('Unexpected large recovery member: '+str(path))
                selected[path.relative_to(PREP).as_posix()]=path
    for path in PREP.iterdir():
        if path.is_file() and path.suffix in ('.py','.sh','.json','.patch') and not path.is_symlink():
            selected[path.name]=path
    if sum(p.stat().st_size for p in selected.values())>300*1024**2:
        raise ValueError('Recovery snapshot exceeds bounded scope')
    files={name:{'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in sorted(selected.items())}
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),
              'mount':subprocess.check_output(['findmnt','-n','-o','FSTYPE,TARGET,SOURCE','-T',str(PREP)],text=True).strip(),
              'installed_packages':sorted([{'name':d.metadata.get('Name'),'version':d.version} for d in importlib.metadata.distributions()],key=lambda d:d['name'] or ''),
              'large_checkpoints_and_model_weights_included':False,
              'shared_ACT_checkpoint_primary_paths':['sft-pinning-validation-20261020-01','pinning-recovery-20261016-04'],
              'environment_rebuild_from_saved_bootstrap_and_package_versions':True,
              'private_keys_API_keys_virtualenv_excluded':True,
              'new_GPU_job_started':False,'full7B_OOM_fix_validated':False}
    manifest={'schema':'jiuge7b-before-job-reset-recovery-v1','files':files,'metadata':metadata}
    output=ROOT/'Jiuge7B-source-dataset-CPU-fixtures-recovery.zip'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for name,path in selected.items():z.write(path,name)
        z.writestr('recovery-manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as z:
        for name,entry in files.items():
            with z.open(name) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256'],name
    result={'schema':'jiuge7b-reset-recovery-verification-v1','passed':True,
            'ACT_bundle':str(output),'bytes':output.stat().st_size,'sha256':sha(output),
            'verified_files':len(files),'uncompressed_bytes':sum(f['bytes'] for f in files.values()),
            'metadata':metadata}
    with (ROOT/'verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
