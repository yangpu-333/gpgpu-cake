"""CPU-only 100-ms lane occupancy export from closed ACT traces, no raw download."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
from analyze_trace import load_trace,union_intervals,interval_length,intersection_length,_gpu_kind

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'sft-paired-tracing-20261020-01'


def main():
    state=json.loads((ROOT/'controller.json').read_text())
    if state['status']!='completed': raise ValueError('Closed capture required')
    bin_us=100000
    rows=[]
    for item in state['cases']:
        case=PREP/'runs'/item['case']
        for window in ('window-01','window-02'):
            for rank in (0,1):
                folder=case/window
                receipt=json.loads((folder/f'rank-{rank}-capture.json').read_text())
                raw=folder/f'rank-{rank}.json.gz'
                if hashlib.sha256(raw.read_bytes()).hexdigest()!=receipt['trace_sha256']:
                    raise ValueError('Trace SHA changed after closure')
                trace,source=load_trace(raw,max_bytes=512*1024**2)
                if source['decompressed_json_sha256']!=receipt['decompressed_json_sha256']:
                    raise ValueError('Decompressed trace SHA differs')
                outer=[e for e in trace['traceEvents'] if e.get('ph')=='X' and e.get('cat')=='user_annotation' and e.get('name')==receipt['window_name']]
                assert len(outer)==1
                start,end=outer[0]['ts'],outer[0]['ts']+outer[0]['dur']
                lanes={name:[] for name in ('CPU optimizer','CPU AdamW math','GPU compute','GPU memcpy','GPU NCCL')}
                directions={name:[] for name in ('HtoD','DtoH')}
                direction_bytes={name:0 for name in directions}
                for e in trace['traceEvents']:
                    if e.get('ph')!='X' or 'ts' not in e or 'dur' not in e: continue
                    a,b=max(start,e['ts']),min(end,e['ts']+e['dur'])
                    if b<=a: continue
                    label=None
                    if e.get('cat')=='user_annotation':
                        label={'jiuge/optimizer_update':'CPU optimizer','jiuge/cpu_adamw_math':'CPU AdamW math'}.get(e.get('name'))
                    if label is None:
                        label={'compute':'GPU compute','collective':'GPU NCCL','memcpy':'GPU memcpy'}.get(_gpu_kind(e))
                    if label: lanes[label].append((a-start,b-start))
                    if e.get('cat')=='gpu_memcpy':
                        for direction in directions:
                            if direction in str(e.get('name','')):
                                directions[direction].append((a-start,b-start))
                                size=e.get('args',{}).get('bytes')
                                if isinstance(size,int) and not isinstance(size,bool): direction_bytes[direction]+=size
                width=end-start
                bins=math.ceil(width/bin_us)
                output={}
                for name,values in lanes.items():
                    data=[0.0]*bins
                    for a,b in union_intervals(values):
                        for k in range(int(a//bin_us),min(bins,math.ceil(b/bin_us))):
                            data[k]+=max(0.0,min(b,(k+1)*bin_us)-max(a,k*bin_us))/min(bin_us,width-k*bin_us)
                    if any(not -1e-9<=v<=1+1e-9 for v in data): raise ValueError('Lane occupancy outside[0,1]')
                    output[name]=[round(v,4) for v in data]
                rows.append({'case':item['case'],'tag':item['tag'],'arm':item['arm'],'rank':rank,'window':window,
                             'window_seconds':width/1e6,'lanes':output,'trace_sha256':receipt['trace_sha256'],
                             'GPU_direction_union_seconds_per_rank_step':{k:interval_length(v)/5e6 for k,v in directions.items()},
                             'CPU_AdamW_overlap_direction_seconds_per_rank_step':{k:intersection_length(lanes['CPU AdamW math'],v)/5e6 for k,v in directions.items()},
                             'direction_recorded_GiB_per_rank_step':{k:v/5/1024**3 for k,v in direction_bytes.items()}})
    record={'schema':'jiuge7b-sft-binned-trace-lanes-v1','utc':datetime.now(timezone.utc).isoformat(),
            'bin_seconds':bin_us/1e6,'rows':rows,'raw_traces':16,'raw_local_download':False,
            'each_panel_time_zero':'its own enclosing capture start, display only; do not infer cross-rank clock alignment',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_trace_SHA_verified':True,'critical_path_proven':False}
    target=PREP/'sft-paired-trace-lanes-20261020.json'
    with target.open('x') as stream: json.dump(record,stream,indent=2)
    print(json.dumps({'path':str(target),'bytes':target.stat().st_size,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))


if __name__=='__main__': main()
