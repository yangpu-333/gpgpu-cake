import hashlib,json,os,time
from pathlib import Path
R=Path(os.environ['JIUGE_DT_ROOT']);C=Path(os.environ['JIUGE_DT_CONTROL']);O=R/'attention-fusion-fullmodel-control-20261028-03';P=R/'CoreX-attention-fusion-20261028-03'
assert os.getuid()==1001 and os.environ['CUDA_VISIBLE_DEVICES']=='' and time.time()<1793204291-1800
old=json.loads((O/'final.json').read_text());assert old['status']=='failed' and 'Insufficient shared budget' in old['error'] and not Path('/proc',str(old['PID'])).exists()

rows=[]
for arm in('native','control','optimized'):
 case=R/f'runs/corex-S2048-attention-audit-{arm}-af03';assert json.loads((case/'case.json').read_text())['exit_code']==0
 for rank in(0,1):
  assert json.loads((case/f'rank-{rank}/entry.json').read_text())['status']=='completed'
  q=P/f'numeric/{arm}/rank-{rank}/all-state-manifest.json';b=q.read_bytes();m=json.loads(b);assert m['step']==3 and m['passed_so_far'] and len(m['tensors'])==3585 and all(x['passed']for x in m['tensors'])
  rows.append({'arm':arm,'rank':rank,'manifest_SHA256':hashlib.sha256(b).hexdigest(),'fields':3585,'max_abs':max(x['max_abs']for x in m['tensors'])})
for name,h in old['source_SHA'].items():assert hashlib.sha256((O/name).read_bytes()).hexdigest()==h
for name in('fullstate_entry.py','fullstate_runner.py','corex_fullstate_reference_v6.py','node0_optimizer_state.py','corex_setup_gradients_v3.py','attention_fusion_adapter.py','fused_scale_mask.py','fused_scale_mask_softmax.py','corex_process_guard_data.py','corex_adamw_local_moments.py'):
 assert (C/name).read_bytes()==(O/name).read_bytes(),name
(C/'numa-plan.json').write_bytes((O/'numa-plan.json').read_bytes())
r={'diagnostic_only_admitted':True,'full_correctness_or_long_loss_or_performance_accepted':False,'prior_native_control_closed_proof':rows,'old_control':str(O),'old_deadline_kept':old['deadline_epoch'],'new_explicit_authorization_start':1793182691,'deadline_epoch':1793204291,'campaign':str(P),'no_GPU_case_rerun':True,'NUMA_free_cutoff_enabled':False}
(C/'full-model-diagnostic-admission.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
