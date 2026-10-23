"""ACT-only NUMA raw delivery; verify every compressed and decompressed trace hash."""
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
CAMPAIGN=PREP/'sft-phased-moments-screen-20261023-07'
ROOT=PREP/'sft-phased-moments-delivery-20261023-07'


def digest(stream):
    h=hashlib.sha256()
    while block:=stream.read(1024*1024): h.update(block)
    return h.hexdigest()


def main():
    state=json.loads((CAMPAIGN/'controller.json').read_text())
    if state['status']!='completed' or state['verified_raw_traces']!=4*len(state['cases']):
        raise ValueError('Complete paired scientific capture required')
    ROOT.mkdir(exist_ok=False)
    selected={}
    traces=[]
    for item in state['cases']:
        folder=PREP/'runs'/item['case']
        proof=json.loads((CAMPAIGN/(item['case']+'-archive-verification.json')).read_text())
        if not proof['passed']: raise ValueError('ACT original archive gate failed')
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.log','.py','.txt','.patch'):
                selected[path.relative_to(PREP).as_posix()]=path
        for window in ('window-01','window-02'):
            for rank in (0,1):
                raw=folder/window/f'rank-{rank}.json.gz'
                receipt=json.loads(raw.with_name(f'rank-{rank}-capture.json').read_text())
                with raw.open('rb') as stream: compressed=digest(stream)
                with gzip.open(raw,'rb') as stream: uncompressed=digest(stream)
                if compressed!=receipt['trace_sha256'] or uncompressed!=receipt['decompressed_json_sha256']:
                    raise ValueError('Original compressed/decompressed trace changed')
                name=raw.relative_to(PREP).as_posix()
                selected[name]=raw
                traces.append({'case':item['case'],'policy':item['policy'],'tag':item['tag'],'rank':rank,'window':window,
                               'steps':receipt['stable_step_ids'],'path_in_bundle':name,
                               'compressed_sha256':compressed,'decompressed_sha256':uncompressed,
                               'NCCL_kernel_events':receipt['gpu_collective_kernel_events']})
    for path in CAMPAIGN.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.log','.py','.sh','.patch'):
            selected[path.relative_to(PREP).as_posix()]=path
    files={}
    for name,path in sorted(selected.items()):
        with path.open('rb') as stream: sha=digest(stream)
        files[name]={'bytes':path.stat().st_size,'sha256':sha}
    manifest={'schema':'jiuge7b-sft-moment-raw-trace-delivery-v1','utc':datetime.now(timezone.utc).isoformat(),
              'files':files,'traces':traces,'cases':state['cases'],
              'scope':'Answer-only full-parameter SFT, accepted pinned repair in every arm; unbound baseline/affinity-only control/CPU-local moment mappings, one seed,35updates per run',
              'hardware':'Exact approved A100 pair for every fresh arm; NODE,sameNUMA;TP2 single node;all prior throughput excluded',
              'raw_traces_included':True,'formal_performance_evidence_in_this_bundle':False,
              'large_model_tensor_or_checkpoint_included':False,'raw_bundle_stays_ACT':True,
              'collector_drop_count_verified':False,'causal_critical_path_proven':False}
    output=ROOT/'Jiuge7B-SFT-S2048-direct-zero-moments-ABC-CBA-traces.zip'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as archive:
        for name,path in selected.items(): archive.write(path,name)
        archive.writestr('trace-delivery-manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        for name,entry in files.items():
            with archive.open(name) as stream:
                if digest(stream)!=entry['sha256']: raise ValueError('Packaged source SHA differs: '+name)
        for row in traces:
            with archive.open(row['path_in_bundle']) as stream,gzip.GzipFile(fileobj=stream) as decompressed:
                if digest(decompressed)!=row['decompressed_sha256']: raise ValueError('Packaged JSON SHA differs')
    with output.open('rb') as stream: sha=digest(stream)
    receipt={'schema':'jiuge7b-sft-moment-trace-delivery-verification-v1','passed':True,
             'bundle':str(output),'bundle_bytes':output.stat().st_size,'bundle_sha256':sha,
             'verified_files':len(files),'verified_raw_traces':len(traces),
             'compressed_and_decompressed_SHA_passed':True,'bundle_downloaded_to_local':False,'traces':traces}
    for name,value in [('manifest.json',manifest),('verification.json',receipt)]:
        with (ROOT/name).open('x') as stream: json.dump(value,stream,indent=2)
    print(json.dumps({k:v for k,v in receipt.items() if k!='traces'}))


if __name__=='__main__': main()
