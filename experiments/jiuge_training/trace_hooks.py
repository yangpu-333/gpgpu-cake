"""Separate whole-step native capture windows; no per-step GPU synchronization."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time


class CaptureWindows:
    def __init__(self, torch, root, rank, case, windows=((21,25),(31,35))):
        self.torch,self.root,self.rank,self.case,self.windows=torch,Path(root),rank,case,windows
        self.profiler=self.outer=None
        self.receipts=[]

    def before(self,step):
        for ordinal,(begin,end) in enumerate(self.windows,1):
            if step != begin:
                continue
            self.window_dir=self.root/f'window-{ordinal:02d}'
            self.window_dir.mkdir(parents=True,exist_ok=True)
            self.window_name=f'jiuge/capture/{self.case}/window-{ordinal:02d}'
            self.torch.cuda.synchronize()
            self.wall_start_ns=time.time_ns()
            self.profiler=self.torch.profiler.profile(activities=[self.torch.profiler.ProfilerActivity.CPU,
                                                                self.torch.profiler.ProfilerActivity.CUDA],
                                                     record_shapes=False,with_stack=False)
            self.profiler.start()
            self.outer=self.torch.profiler.record_function(self.window_name)
            self.outer.__enter__()

    def range(self,step):
        return self.torch.profiler.record_function(f'jiuge/train_step/{step}')

    def after(self,step):
        for ordinal,(begin,end) in enumerate(self.windows,1):
            if step != end or self.profiler is None:
                continue
            self.torch.cuda.synchronize()
            self.outer.__exit__(None,None,None)
            self.profiler.stop()
            self.wall_end_ns=time.time_ns()
            path=self.window_dir/f'rank-{self.rank}.json.gz'
            self.profiler.export_chrome_trace(str(path))
            with gzip.open(path,'rb') as stream:
                data=stream.read()
            trace=json.loads(data)
            kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel']
            nccl=[e for e in kernels if 'nccl' in e.get('name','').lower()]
            launches=[e for e in trace['traceEvents'] if e.get('cat') in ('cuda_runtime','cuda_driver')
                      and 'launchkernel' in e.get('name','').lower()]
            kernel_ids={e.get('args',{}).get('correlation') for e in kernels}
            launch_ids={e.get('args',{}).get('correlation') for e in launches}
            record={'schema':'jiuge7b-native-window-capture-v1','utc':datetime.now(timezone.utc).isoformat(),
                    'rank':self.rank,'world_size':2,'case':self.case,'window':ordinal,'window_name':self.window_name,
                    'stable_step_ids':list(range(begin,end+1)), 'window_start_wall_ns':self.wall_start_ns,
                    'window_end_wall_ns':self.wall_end_ns,'base_time_nanoseconds':trace.get('baseTimeNanoseconds'),
                    'trace_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'decompressed_json_sha256':hashlib.sha256(data).hexdigest(),
                    'gpu_kernel_events':len(kernels),'gpu_collective_kernel_events':len(nccl),
                    'recorded_kernel_launches':len(launches),
                    'launch_ids_without_gpu_kernel':sorted(str(x) for x in launch_ids-kernel_ids),
                    'whole_window_boundary_drained':True,'per_step_extra_synchronization':False,
                    'per_operator_extra_synchronization':False,'record_shapes':False,'with_stack':False,
                    'reported_collector_drop_count':None,
                    'collector_drop_count_not_exposed_by_torch_api':True,
                    'formal_throughput_evidence':False}
            with (self.window_dir/f'rank-{self.rank}-capture.json').open('x') as stream:
                json.dump(record,stream,indent=2)
            self.receipts.append(record)
            self.profiler=self.outer=None
            print(json.dumps({'capture_complete':ordinal,'rank':self.rank,'kernel_events':len(kernels),
                              'collective_kernel_events':len(nccl)}),flush=True)

    def close(self):
        if self.profiler is not None:
            self.torch.cuda.synchronize()
            self.outer.__exit__(None,None,None)
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.window_dir/f'rank-{self.rank}-partial.json.gz'))
            self.profiler=None
