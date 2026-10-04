"""Untimed script availability/graph evidence; no optimization math or GPU calls."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--version',required=True);a=p.parse_args()
base=Path('/tmp/kda-attention-stage31');source=base/('candidate-'+a.version+'.py')
spec=importlib.util.spec_from_file_location('attention_jit_'+a.version,source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
rows=[]
for name in ('_fwd_scores','_fwd_context','_bwd_context','_bwd_scores'):
 fn=getattr(m,name);scripted=isinstance(fn,torch.jit.ScriptFunction)
 code=str(fn.code) if scripted else None;graph=str(fn.graph) if scripted else None
 rows.append({'name':name,'type':type(fn).__name__,'scripted':scripted,
 'code':code,'graph':graph,'graph_sha256':hashlib.sha256(graph.encode()).hexdigest() if graph else None})
r={'schema':'untimed-attention-script-availability-v1','version':a.version,
 'candidate_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'torch':torch.__version__,'gpu_calls':False,'rows':rows,
 'passed':all(x['scripted'] for x in rows),
 'scope':'Import-time ScriptFunction and static graph availability only; not actual GPU timeline or performance proof.'}
out=base/('jit-availability-'+a.version+'.json')
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'passed':r['passed'],'output':str(out)}))
if not r['passed']:raise SystemExit(1)
