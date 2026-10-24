"""CPU-only completion of frozen semantic/raw gates after a false resume counter gate."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from review_sft_phased_resume_policy_20261024 import PREP,ROOT,review,read,sha

DEST=PREP/'sft-phased-correctness-postprocess-20261024-03'
STOP=1792853953


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    DEST.mkdir(exist_ok=False)
    with (DEST/'postprocess.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'GPU_used':False,'formal_performance_started':False,'formal_performance_passed':False,'completed_cases':read(ROOT/'controller.json')['completed_cases'],'commands':[],'pid':os.getpid(),'stop_before_epoch':STOP}
        def save(status,**values):
            state.update(status=status,updated_utc=datetime.now(timezone.utc).isoformat(),**values)
            p=DEST/'controller.json.tmp';p.write_text(json.dumps(state,indent=2));p.replace(DEST/'controller.json')
        def command(name,args):
            assert time.time()<STOP-300
            with (DEST/(name+'.stdout.log')).open('x') as out,(DEST/(name+'.stderr.log')).open('x') as err:
                child=subprocess.Popen([sys.executable,*map(str,args)],cwd=PREP,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
                save(name,active_child_pid=child.pid)
                while child.poll() is None:
                    if time.time()>STOP-180:
                        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=15);raise TimeoutError('Preserve original authorized deadline')
                    save(name,active_child_pid=child.pid);time.sleep(15)
                state.pop('active_child_pid',None);state['commands'].append({'name':name,'exit_code':child.returncode})
                if child.returncode:raise RuntimeError(name+' failed')
        try:
            save('review-resume-policy')
            policy=review()
            with (DEST/'resume-policy-review.json').open('x') as f:json.dump(policy,f,indent=2)
            command('verify-semantics',[PREP/'verify_sft_pinning_semantics.py','--root',ROOT/'semantics','--resume-campaign',ROOT/'numeric','--output',DEST/'semantics-verification.json'])
            save('checkpoint-SHA')
            checkpoint=ROOT/'checkpoints/native-step2';identities=[]
            for path in sorted(checkpoint.rglob('*')):
                if not path.is_file():continue
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for b in iter(lambda:f.read(8*1024**2),b''):
                        if time.time()>STOP-180:raise TimeoutError('Original deadline during SHA')
                        h.update(b)
                identities.append({'file':str(path.relative_to(checkpoint)),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
            with (DEST/'checkpoint-sha256.json').open('x') as f:json.dump({'checkpoint':str(checkpoint),'files':identities},f,indent=2)
            command('verify-all-originals',[PREP/'verify_pinning_archive.py','--root',ROOT/'numeric','--include-resume','--output',DEST/'primary-archive-verification.json'])
            primary=read(DEST/'primary-archive-verification.json');sem=read(DEST/'semantics-verification.json')
            assert primary['passed'] and primary['checkpoint_resume_tested'] and len(primary['runs'])==10 and sem['passed'] and sem['full_model_native_checkpoint_resume_verified']
            files=[ROOT/'controller.json',ROOT/'reference-origins.json',PREP/'sft-phased-acceptance-source-manifest-20261024-02.json']
            for arm in ('native','control','optimized','resume-restored','resume-next-step'):
                for rank in (0,1):
                    d=ROOT/'numeric'/arm/f'rank-{rank}';files.append(d/'all-state-manifest.json')
                    if (d/'completion.json').exists():files.append(d/'completion.json')
            files.extend(ROOT/'numeric'/f'resume-rank-{r}-completion.json' for r in (0,1))
            for name in ('resume-policy-review.json','semantics-verification.json','checkpoint-sha256.json','primary-archive-verification.json'):
                files.append(DEST/name)
            for name in ('phased-resume-native-20261024-02','phased-resume-optimized-20261024-02'):
                case=PREP/'runs'/name;files.extend(case/x for x in ('case.json','stdout.log','stderr.log'))
                for rank in (0,1):
                    files.extend([case/f'rank-{rank}/entry.json',case/f'rank-{rank}/checkpoint-compat.json',case/f'sft-objective-rank-{rank}.json'])
                    for pattern in (f'phased-policy-rank-{rank}.json',f'audit-cache-rank-{rank}.json'):
                        if (case/pattern).exists():files.append(case/pattern)
            assert all(f.is_file() for f in files) and sum(f.stat().st_size for f in files)<64*1024**2
            bundle=DEST/'correctness-key-results.zip';catalog={}
            with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
                for f in files:
                    name=str(f.relative_to(PREP));b=f.read_bytes();catalog[name]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};z.writestr(name,b)
                z.writestr('key-results-manifest.json',json.dumps({'schema':'jiuge-pinning-key-results-v1','primary_root':str(PREP),'files':catalog,'key_uncompressed_bytes':sum(x['bytes'] for x in catalog.values()),'full_raw_originals_included':False,'full_raw_local_copy_verified':False}))
            from pinning_key_results import verify
            verify(bundle,DEST/'ACT-key-verification.json',64*1024**2)
            with (DEST/'final-key-bundle.json').open('x') as f:json.dump({'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':sha(bundle)},f)
            save('completed',full_correctness_resume_and_primary_archive_passed=True,formal_not_started_reason='less than original3h formal reserve; no GPU restart',raw_originals_location='ACT Ceph',raw_chunks=primary['unique_raw_chunks_verified'])
        except BaseException as error:
            save('stopped',error=repr(error));raise


if __name__=='__main__':main()
