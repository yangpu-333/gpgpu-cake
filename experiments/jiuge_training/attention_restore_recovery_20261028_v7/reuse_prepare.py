from pathlib import Path
import hashlib,json,os,time
R=Path(os.environ['JIUGE_DT_ROOT']);C=Path(os.environ['JIUGE_DT_CONTROL']);O=R/'attention-fusion-CPU-review-control-20261028-01';P=R/'CoreX-attention-fusion-20261028-03';a=json.loads((C/'authorization.json').read_text());proof=a['fullstate_independent_admission'];assert proof['three_arm_numerical_and_SFT_semantics_passed']and proof['local_independent_review_passed']and proof['both_ends_key_ZIP_SHA_verified'];assert json.loads((O/'final.json').read_text())['status']=='failed';assert not Path('/proc',str(json.loads((O/'final.json').read_text())['PID'])).exists()
for arm in('native','control','optimized'):
 assert json.loads((R/f'runs/corex-S2048-attention-audit-{arm}-af03/case.json').read_text())['exit_code']==0
 for rank in(0,1):assert hashlib.sha256((P/f'numeric/{arm}/rank-{rank}/all-state-manifest.json').read_bytes()).hexdigest()==proof['manifest_SHA256'][arm+'/'+str(rank)]
for name in('fullstate_entry.py','fullstate_runner.py','corex_setup_gradients_v3.py','node0_optimizer_state.py','corex_adamw_local_moments.py','attention_fusion_adapter.py','fused_scale_mask.py'):assert hashlib.sha256((C/name).read_bytes()).hexdigest()==proof['source_SHA256'][name]
b=(C/'guard_data_original.py').read_bytes();assert hashlib.sha256(b).hexdigest()==proof['source_SHA256']['corex_process_guard_data.py'];assert hashlib.sha256((C/'corex_process_guard_data.py').read_bytes()).hexdigest()==a['sources']['corex_process_guard_data.py']['sha256']
(C/'numa-plan.json').write_bytes((O/'numa-plan.json').read_bytes());(C/'strict-fullstate-independent-admission.json').write_text(json.dumps(proof,indent=2));(C/'full-model-diagnostic-admission.json').write_text(json.dumps({'diagnostic_only_admitted':True,'full_correctness_or_long_loss_or_performance_accepted':False,'three_arm_proof_reused_without_rerun':True,'memory_policy':'actual kernel144GiB DATA and own292GiB RSS retained; remove arbitrary per-process8GiB+2allocation stop; no NUMA cutoff','deadline_epoch':1793204291}));print('Independent numerical proof and source bindings retained; only resource preemption changed')

O6=R/'attention-native-restore-control-20261028-06'
prior=json.loads((O6/'checkpoint-before-resume-SHA.json').read_text())
for row in prior['files']:
 path=Path(row['file']);assert path.is_relative_to(P/'checkpoints-rs04');st=path.stat()
 assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])
(C/'checkpoint-before-resume-SHA.json').write_text(json.dumps({**prior,'full_SHA_reused_from':str(O6),'metadata_identity_rechecked':True},indent=2))
