"""Strict scalar and input equivalence probe; not formal training acceptance."""
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--native-root',type=Path,required=True)
    p.add_argument('--shapes',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    import torch
    torch.cuda.set_device(0)
    source=a.native_root/'megatron/core/utils.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='local_multi_tensor_l2_norm')
    ns={'torch':torch}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    native=ns[node.name]
    spec=importlib.util.spec_from_file_location('gradnorm_candidate',a.candidate)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    candidate=module.local_multi_tensor_l2_norm
    shapes=json.loads(a.shapes.read_text())
    result={'scope':'Standalone scalar/input probe and synchronized wall microbenchmark; not full training or promotion.',
      'candidate_sha256':hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
      'native_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
      'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'rows':[]}
    g=torch.Generator().manual_seed(26101)
    flag=torch.zeros(1,device='cuda',dtype=torch.int32)
    for scope,shape_list in shapes.items():
        for scale in (0.0,0.001,1.0,100.0):
            grads=[(torch.randn(shape,generator=g)*scale).to('cuda') for shape in shape_list]
            before=[x.cpu().clone() for x in grads]
            n,_=native(2048,flag,[grads],False)
            o,aux=candidate(2048,flag,[grads],False,native=native)
            exact=torch.equal(n.cpu(),o.cpu())
            unchanged=all(torch.equal(x.cpu(),b) for x,b in zip(grads,before))
            row={'scope':scope,'scale':scale,'tensor_count':len(grads),'elements':sum(x.numel() for x in grads),
                 'native_norm':float(n.cpu()[0]),'candidate_norm':float(o.cpu()[0]),
                 'bitwise_equal':exact,'inputs_unchanged':unchanged,'auxiliary_none':aux is None}
            if exact and unchanged and aux is None:
                for _ in range(3):
                    native(2048,flag,[grads],False);candidate(2048,flag,[grads],False,native=native)
                samples={'native':[],'candidate':[]}
                for trial in range(9):
                    order=('native','candidate') if trial%2==0 else ('candidate','native')
                    for arm in order:
                        torch.cuda.synchronize();start=time.perf_counter()
                        (native(2048,flag,[grads],False) if arm=='native' else candidate(2048,flag,[grads],False,native=native))
                        torch.cuda.synchronize();samples[arm].append((time.perf_counter()-start)*1000)
                row['synchronized_wall_ms']=samples
            result['rows'].append(row)
            print(json.dumps({k:row[k] for k in ('scope','scale','bitwise_equal','inputs_unchanged')}),flush=True)
    result['status']='passed' if all(r['bitwise_equal'] and r['inputs_unchanged'] and r['auxiliary_none'] for r in result['rows']) else 'rejected'
    a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        main()
