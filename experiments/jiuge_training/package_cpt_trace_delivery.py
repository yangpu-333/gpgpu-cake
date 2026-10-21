"""Package already completed CPT traces on ACT; download only the small receipt."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'tracing-delivery-20261017-01'
CASES=('pinning-native-profile-01','pinning-optimized-profile-01')


def digest(stream):
    value=hashlib.sha256()
    while block:=stream.read(1024*1024):
        value.update(block)
    return value.hexdigest()


def main():
    ROOT.mkdir(exist_ok=False)
    selected={}
    traces=[]
    cases=[]
    for name in CASES:
        folder=PREP/'runs'/name
        case=json.loads((folder/'case.json').read_text())
        if case['exit_code']!=0:
            raise ValueError('Only completed native CPT profiles may be packaged')
        for rank in range(2):
            entry=json.loads((folder/f'rank-{rank}/entry.json').read_text())
            if entry['status']!='completed' or entry['completed_native_steps']!=[
                {'step':i,'skipped':0} for i in range(1,36)]:
                raise ValueError('Actual 35-step complete full-model training required')
            for filename in ('entry.json','weight-loading.json','optimizer-state.json'):
                path=folder/f'rank-{rank}'/filename
                if path.is_file():
                    selected[path.relative_to(PREP).as_posix()]=path
        for filename in ('case.json','case-start.json','stdout.log','stderr.log','trace-analysis.json',
                         'communication-analysis.json','archive-manifest.json'):
            path=folder/filename
            selected[path.relative_to(PREP).as_posix()]=path
        for path in (folder/'source').glob('*.py'):
            selected[path.relative_to(PREP).as_posix()]=path
        for window in (1,2):
            for rank in range(2):
                path=folder/f'window-{window:02d}/rank-{rank}.json.gz'
                receipt_path=path.with_name(f'rank-{rank}-capture.json')
                receipt=json.loads(receipt_path.read_text())
                with path.open('rb') as stream:
                    compressed=digest(stream)
                with gzip.open(path,'rb') as stream:
                    uncompressed=digest(stream)
                if compressed!=receipt['trace_sha256'] or uncompressed!=receipt['decompressed_json_sha256']:
                    raise ValueError('Actual trace hash differs from original capture receipt')
                if not receipt['whole_window_boundary_drained'] or not receipt['gpu_collective_kernel_events']:
                    raise ValueError('Whole-window GPU/NCCL evidence required')
                for source in (path,receipt_path):
                    selected[source.relative_to(PREP).as_posix()]=source
                traces.append({'case':name,'window':window,'rank':rank,'steps':receipt['stable_step_ids'],
                    'path_in_bundle':path.relative_to(PREP).as_posix(),'compressed_sha256':compressed,
                    'decompressed_sha256':uncompressed,'kernel_events':receipt['gpu_kernel_events'],
                    'NCCL_kernel_events':receipt['gpu_collective_kernel_events']})
        cases.append({'case':name,'steps_each_rank':35,'sequence_length':2048,'actual_full_parameter_CPT':True})
    files={}
    for name,path in sorted(selected.items()):
        with path.open('rb') as stream:
            files[name]={'bytes':path.stat().st_size,'sha256':digest(stream)}
    manifest={'schema':'jiuge7b-CPT-trace-delivery-manifest-v1',
              'utc':datetime.now(timezone.utc).isoformat(),'files':files,'traces':traces,'cases':cases,
              'hardware':'2 NVIDIA A100 on inspur-gpu-04; original Oct16 allocation',
              'scope':'native/pinned-repair stable whole-CPT-step diagnostics, not SFT',
              'formal_performance_evidence_in_this_bundle':False,'SFT_training_completed':False,
              'raw_traces_included':True,'large_numeric_states_or_model_weights_included':False,
              'collector_drop_count_available':False,'causal_critical_path_proven':False,
              'source_parameter_count':7668831744,'raw_bundle_remains_on_ACT':True}
    output=ROOT/'Jiuge7B-CPT-S2048-native-and-pinned-traces.zip'
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as archive:
        for name,path in selected.items():
            archive.write(path,name)
        archive.writestr('trace-delivery-manifest.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(output) as archive:
        for name,record in files.items():
            with archive.open(name) as stream:
                if digest(stream)!=record['sha256']:
                    raise ValueError('Packaged file SHA differs: '+name)
        for trace in traces:
            with archive.open(trace['path_in_bundle']) as stream, gzip.GzipFile(fileobj=stream) as decompressed:
                if digest(decompressed)!=trace['decompressed_sha256']:
                    raise ValueError('Packaged decompressed trace SHA differs')
    with output.open('rb') as stream:
        bundle_sha=digest(stream)
    receipt={'schema':'jiuge7b-CPT-trace-delivery-verification-v1','passed':True,
             'utc':datetime.now(timezone.utc).isoformat(),'bundle':str(output),'bundle_bytes':output.stat().st_size,
             'bundle_sha256':bundle_sha,'verified_files':len(files),'raw_trace_windows':len(traces),
             'all_compressed_and_decompressed_trace_hashes_verified':True,'traces':traces,
             'bundle_downloaded_to_local':False,'SFT_training_completed':False,
             'visualization':'Unzip on ACT or selected destination; gunzip one rank/window JSON and open in Perfetto UI. Preserve both rank clock bases.'}
    for name,value in (('manifest.json',manifest),('verification.json',receipt)):
        with (ROOT/name).open('x') as stream:
            json.dump(value,stream,indent=2)
    print(json.dumps({key:value for key,value in receipt.items() if key!='traces'}),flush=True)


if __name__=='__main__':
    main()
