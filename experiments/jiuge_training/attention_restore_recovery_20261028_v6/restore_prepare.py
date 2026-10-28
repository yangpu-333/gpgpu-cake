"""CPU interlock: isolated restore backends only after fresh independent numerical/SFT admission."""
import hashlib,json,os,time,sys
from pathlib import Path
root=Path(os.environ['JIUGE_DT_ROOT']);control=Path(os.environ['JIUGE_DT_CONTROL']);sys.path[:0]=[str(control),str(root)]
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.getuid()==1001 and time.time()<1793204291-1800
camp=root/'CoreX-attention-fusion-20261028-03'
proof_path=control/'strict-fullstate-independent-admission.json';proof=json.loads(proof_path.read_text())
assert proof['three_arm_numerical_and_SFT_semantics_passed'] is True
assert proof['campaign']==str(camp) and proof['native_checkpoint_resume_accepted'] is False
assert proof['deadline_epoch']==1793204291 and proof['local_independent_review_passed'] is True
for name in ('fullstate_entry.py','fullstate_runner.py','corex_fullstate_reference_v6.py','corex_setup_gradients_v3.py','node0_optimizer_state.py','corex_adamw_local_moments.py','attention_fusion_adapter.py','fused_scale_mask.py'):
 assert hashlib.sha256((control/name).read_bytes()).hexdigest()==proof['source_SHA256'][name]
rows=[]
for arm in ('native','control','optimized'):
 for rank in (0,1):
  path=camp/'numeric'/arm/f'rank-{rank}'/'all-state-manifest.json';v=json.loads(path.read_text());assert v['step']==3 and v['passed_so_far'] and len(v['tensors'])==3585 and all(x['passed'] for x in v['tensors'])
  digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==proof['manifest_SHA256'][f'{arm}/{rank}'];rows.append({'arm':arm,'rank':rank,'manifest_SHA256':digest})
assert (camp/'checkpoints-rs04/iter_0000002/.metadata').is_file()
from pinning_candidate import prepare
paths=[]
for stage,arm in (('checkpoint','native'),('resume','native'),('resume','optimized')):
 path=control/'pinning-backends'/f'{stage}-{arm}';assert not path.exists();prepare(path);paths.append(str(path))
result={'native_restore_CPU_preparation_passed':True,'GPU_started':False,'native_checkpoint_resume_accepted':False,'full_correctness_or_performance_accepted':False,'independent_admission_SHA256':hashlib.sha256(proof_path.read_bytes()).hexdigest(),'rows':rows,'isolated_backends':paths,'campaign':str(camp),'deadline_epoch':1793204291,'same_default_GPU_mode_and_common_node0_setup_required':True}
with (control/'strict-native-restore-CPU-admission.json').open('x') as stream:json.dump(result,stream,indent=2)
print(json.dumps(result),flush=True)
