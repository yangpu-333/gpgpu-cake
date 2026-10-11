"""Prepare exact pinned runtime bytes and an independently frozen recovery collection."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE=Path(__file__).resolve().parent
BF=HERE.parent
OLD=BF/'moe_scale'
FORMAL=BF.parent/'megatron_training_entry'
COMMIT='5be9626709af2722333bf54797c954c09edeada3'
REMOTE='/private/kda-biv150-stage35-recovery01'
sha=lambda data:hashlib.sha256(data).hexdigest()
read=lambda path:json.loads(Path(path).read_text(encoding='utf-8'))


def new(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        assert path.is_file() and path.read_bytes()==data, 'existing immutable bytes differ: '+str(path)
        return
    with path.open('xb') as stream:stream.write(data)


def main():
    assert not (HERE/'evidence/runtime-archive.json').exists(),'complete immutable archive already exists'
    frozen=read(OLD/'models/controllers.json')
    for name,digest in frozen['files'].items():
        data=(OLD/'models'/name).read_bytes();assert sha(data)==digest
        new(HERE/'models'/name,data)
    plan=read(OLD/'recovery-plan.json')
    plan.update(status='frozen_before_fresh_model_training',remote_models_root=REMOTE+'/models',
                native_runtime_storage='Reconstructible pinned source files at the original /tmp paths; original raw captures/checkpoints at verified /private NFS paths',
                source_controller_files_byte_identical=25)
    new(HERE/'plan.json',(json.dumps(plan,indent=2)+'\n').encode())
    provenance={'schema':'stage35-recovery01-controller-provenance-v1',
        'parent_controllers_sha256':sha((OLD/'models/controllers.json').read_bytes()),
        'pod_replacement_record_sha256':sha((OLD/'evidence/pod-replacement-20261009.json').read_bytes()),
        'all25_model_controller_files_byte_identical':True,'retuned':False,
        'scope':'New collection namespace and raw backup before timing; mathematical gates, defaults and native timer unchanged.'}
    new(HERE/'provenance.json',(json.dumps(provenance,indent=2)+'\n').encode())
    frozen.update(plan_sha256=sha((HERE/'plan.json').read_bytes()),provenance_sha256=sha((HERE/'provenance.json').read_bytes()))
    new(HERE/'models/controllers.json',(json.dumps(frozen,indent=2)+'\n').encode())
    for name in ('verify_receipts.py','test_verify_receipts.py'):
        data=(OLD/name).read_bytes()
        if name=='verify_receipts.py':
            text=data.decode().replace("REMOTE='/tmp/kda-moe-scale-stage35/'","REMOTE="+repr(REMOTE+'/models/'))
            text=text.replace("CONTROLLERS_SHA='032bb6ab21bee2ff24c74c64c91cb578c442ffee7e44b36fab96a8f660830de6'",
                              'CONTROLLERS_SHA='+repr(sha((HERE/'models/controllers.json').read_bytes())))
            data=text.encode()
        new(HERE/name,data)
    manifest=read(FORMAL/'source_manifest.json');assert manifest['commit']==COMMIT and len(manifest['files'])==600
    source=BF.parents[2]/'research/code/megatron-lm-core-v0.19.0'
    names=list(manifest['files'])
    response=subprocess.run(['git','-C',str(source),'cat-file','--batch'],
        input=''.join(COMMIT+':'+name+'\n' for name in names).encode(),capture_output=True,check=True)
    stream=io.BytesIO(response.stdout);payload={}
    for name in names:
        fields=stream.readline().split();assert len(fields)==3 and fields[1]==b'blob'
        data=stream.read(int(fields[2]));assert stream.read(1)==b'\n' and sha(data)==manifest['files'][name]
        payload['native/'+name]=data
    assert not stream.read();payload['native/.git/HEAD']=(COMMIT+'\n').encode()
    selected={
        'bf16':(BF,('compare_training.py','summarize_performance.py','launch_training.py','kernel_adapter.py','run_case.py','candidates/0008/candidate.py')),
        'formal':(FORMAL,('runtime_compat.py','source_manifest.json','summarize_performance.py')),
        'gradnorm':(BF/'evidence/gradnorm-stage26',('launch_gradnorm-04.py',)),
    }
    for group,(folder,relatives) in selected.items():
        for relative in relatives:payload[group+'/'+relative]=(folder/relative).read_bytes()
    payload['gradnorm/candidate.py']=(BF/'gradnorm_candidates/001/candidate.py').read_bytes()
    for name in frozen['files']:payload['experiment/models/'+name]=(HERE/'models'/name).read_bytes()
    for name in ('models/controllers.json','plan.json','provenance.json'):
        payload['experiment/'+name]=(HERE/name).read_bytes()
    rows=[{'path':name,'bytes':len(data),'sha256':sha(data)} for name,data in sorted(payload.items())]
    bundle_manifest={'schema':'stage35-recovery01-runtime-bundle-v1','megatron_commit':COMMIT,'native_sources':600,
        'files':rows,'model_controller_files_byte_identical':25,
        'controllers_manifest_sha256':sha((HERE/'models/controllers.json').read_bytes()),
        'credentials_included':False,'installed_packages_copied':False,'gpu_work_executed':False,
        'remote_models_root':REMOTE+'/models','packager_sha256':sha(Path(__file__).read_bytes())}
    new(HERE/'evidence/runtime-bundle-manifest.json',(json.dumps(bundle_manifest,indent=2)+'\n').encode())
    output=HERE/'results/runtime-recovery01.zip';output.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,data in sorted(payload.items()):archive.writestr(name,data)
        archive.writestr('manifest.json',json.dumps(bundle_manifest,indent=2)+'\n')
    new(HERE/'evidence/runtime-archive.json',(json.dumps({'path':output.name,'bytes':output.stat().st_size,
        'sha256':sha(output.read_bytes()),'manifest_sha256':sha((HERE/'evidence/runtime-bundle-manifest.json').read_bytes())},indent=2)+'\n').encode())
    print(json.dumps({'files':len(rows),'bytes':sum(r['bytes'] for r in rows),'archive_bytes':output.stat().st_size,
        'archive_sha256':sha(output.read_bytes()),'controllers_sha256':bundle_manifest['controllers_manifest_sha256']}))


if __name__=='__main__':main()
