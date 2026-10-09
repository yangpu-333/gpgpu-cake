"""Seal three frozen MoE comparisons and candidate-specific independent gates."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
a=argparse.ArgumentParser();a.add_argument('--version',required=True);v=a.parse_args().version
out=ROOT/('audit-'+v+'.json');assert not out.exists()
candidate=ROOT/'candidates'/v/'candidate.py'
probe=read(ROOT/('probe-'+v+'.json'));offset=read(ROOT/('offset-'+v+'.json'))
assert probe['status']=='passed' and offset['passed'] is True
assert probe['candidate_sha256']==offset['candidate_sha256']==sha(candidate)
assert probe['verifier_sha256']==sha(ROOT/'probe.py') and offset['verifier_sha256']==sha(ROOT/'probe_offset.py')
comparisons={}
for name,arms in [('comparison-admission01.json',('native','control')),('comparison-native-'+v+'.json',('native','optimized')),('comparison-control-'+v+'.json',('control','optimized'))]:
 value=read(ROOT/name);assert value['passed'] is True and (value['left_arm'],value['right_arm'])==arms
 assert value['stage34_comparator_sha256']==sha(ROOT/'compare.py')
 assert len(value['checks'])==2627 and len(value['tensor_comparisons'])==1289 and len(value['route_input_files'])==48
 for row in value['input_files']+value['route_input_files']:
  path=Path(row['path']);assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
 comparisons[name]={'path':name,'sha256':sha(ROOT/name),'check_count':len(value['checks']),'tensor_count':len(value['tensor_comparisons'])}
r={'schema':'moe-weighted-stage34-sealed-audit-v1','passed':True,'candidate_sha256':sha(candidate),'contract_sha256':sha(ROOT/'contract.json'),
 'controller_sha256':{n:sha(ROOT/n) for n in ('launch_weighted.py','run_case.py','compare.py','probe.py','probe_offset.py','seal_audit.py','run_performance.py')},
 'probe_sha256':sha(ROOT/('probe-'+v+'.json')),'offset_probe_sha256':sha(ROOT/('offset-'+v+'.json')),'comparisons':comparisons,
 'ce_sha256':sha(Path('/tmp/kda-bf16-continuation/megatron_bf16/candidates/0008/candidate.py')),
 'gradnorm_sha256':sha(Path('/tmp/kda-gradnorm-stage26/candidate.py'))}
assert r['ce_sha256']=='9b22e12b4744b2a41b9209faa280e361e6defe1d5480f5e5683fddd247b7f493'
assert r['gradnorm_sha256']=='84c851ce5af7a49ba97685710790146d2c315f9c8837259366bffd7831f67dc0'
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
