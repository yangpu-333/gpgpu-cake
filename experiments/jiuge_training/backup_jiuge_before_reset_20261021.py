"""Preserve the useful bounded raw trace deliveries locally before job reset."""
from datetime import datetime,timezone
import hashlib
import json
import msvcrt
from pathlib import Path
import subprocess
from drive_sft_pinning_validation import SCP,PREP

BASE=Path(__file__).resolve().parent
ROOT=BASE/'results/reset-backup-20261021'
PACKS=[
 ('sft-moment-screen-failure-20261021-01/Jiuge7B-SFT-local-moments-interrupted-OOM-evidence.zip',73324713,'9d93d18e8136ab1ce6fd0e371249aec9874ed835a0b3bac2b5d0290b7639a8b0'),
 ('sft-paired-trace-delivery-20261020-01/Jiuge7B-SFT-S2048-native-and-pinned-ABBA-traces.zip',136655848,'702a18531f5d5325482ebe609fbe3a1117b58cd2a57b7859dcbe27a77bdcc5bc'),
 ('sft-numa-delivery-20261021-02/Jiuge7B-SFT-S2048-pinned-NUMA-v2-traces.zip',217181114,'f8b5580d224b030bfe1ae7c83f65e19cfff517e4ca92dd85f1a56571354a4684'),
 ('tracing-delivery-20261017-02/Jiuge7B-native-CPT-SFT-S2048-traces.zip',67633445,'d7603aa99887f5c8eab117a8b4f4f42c553b0f6278a62c0a9c921d5187872921'),
 ('tracing-delivery-20261017-01/Jiuge7B-CPT-S2048-native-and-pinned-traces.zip',67939630,'741169a78b1ca2f24127cf5d57f2b617f5867e0616df7e1be923d06aede0ec9b'),
]


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    with (ROOT/'backup.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'1');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        rows=[]
        def state(status,**extra):
            value={'status':status,'updated_utc':datetime.now(timezone.utc).isoformat(),
                   'verified':rows,'large_training_checkpoints_included':False,**extra}
            tmp=ROOT/'status.tmp';tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(ROOT/'status.json')
        for remote,size,expected in PACKS:
            target=ROOT/Path(remote).name
            state('downloading',active=remote)
            if not target.exists():
                partial=target.with_suffix('.zip.partial')
                subprocess.run([*SCP,'huids25@192.168.5.200:'+PREP+'/'+remote,str(partial)],
                               check=True,timeout=3600)
                if partial.stat().st_size!=size or sha(partial)!=expected:
                    raise ValueError('Downloaded raw delivery size/SHA differs: '+remote)
                partial.replace(target)
            if target.stat().st_size!=size or sha(target)!=expected:
                raise ValueError('Existing closed raw backup differs')
            rows.append({'ACT':PREP+'/'+remote,'local':str(target),'bytes':size,'sha256':expected})
            print(json.dumps({'verified':target.name,'bytes':size}),flush=True)
        state('completed_trace_delivery_backup',verified_total_bytes=sum(x['bytes'] for x in rows),
              checkpoint_originals_still_only_ACT=True)


if __name__=='__main__':main()
