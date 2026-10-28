"""Bounded independent numerical/SFT key review, only after all real GPU/CPU stages.

Does not claim native save/resume, long loss or performance. Originals stay DT.
"""
import hashlib,io,json,shlex,subprocess,time,zipfile
from pathlib import PurePosixPath
from inspect_datathinking_biv150_20261025 import BASE,AUDIT,SSH
ROOT='/share/jiuge7b-corex-validation-20261025-01'
CONTROL='attention-fusion-CPU-review-control-20261028-01'
CAMP='CoreX-attention-fusion-20261028-03'
REMOTE=r'''
import hashlib,io,json,sys,time,zipfile
from pathlib import Path
r=Path('/share/jiuge7b-corex-validation-20261025-01');c=r/'attention-fusion-CPU-review-control-20261028-01';camp=r/'CoreX-attention-fusion-20261028-03'
v=json.loads((c/'controller.json').read_text());assert v['status']=='awaiting-reviewed-stage-plan' and v['next_plan_sequence']==2 and not v.get('current_stage')
assert len(v['stages'])==3 and all(x['exit_code']==0 and not x.get('owned_survivors_after_parent_exit') for x in v['stages'])
assert json.loads((c/'followup-incremental-raw-verification.json').read_text())['passed']
assert json.loads((c/'followup-fresh-SFT-source-semantic-verification.json').read_text())['passed']
auth=json.loads((c/'authorization.json').read_text())
for name,h in auth['original179'].items():
 b=(r/'original-sources'/name).read_bytes();assert hashlib.sha256(b).hexdigest()==h
 derived=b.replace(b'/home/huids25/kda-jiuge-prep-20261014',str(r).encode())
 if name in ('run_sft_pinning_case.py','run_pinning_case.py'):
  runner=(r/'original-sources/run_native_case.py').read_bytes().replace(b'/home/huids25/kda-jiuge-prep-20261014',str(r).encode())
  derived=derived.replace(auth['original179']['run_native_case.py'].encode(),hashlib.sha256(runner).hexdigest().encode())
 assert (r/name).read_bytes()==derived
native=json.loads((r/'source_manifest.json').read_text())['files'];assert len(native)==600
for name,h in native.items():assert hashlib.sha256((r/'megatron-native-5be9626'/name).read_bytes()).hexdigest()==h
for name,row in auth['sources'].items():assert hashlib.sha256((c/name).read_bytes()).hexdigest()==row['sha256']
for row in v['stages']:assert hashlib.sha256((c/row['script']).read_bytes()).hexdigest()==row['source_SHA256']
with (c/'fullstate-SFT-source-postcheck.json').open('x') as f:json.dump({'passed':True,'original179_unchanged':True,'live_original179_exact_path_derivation_unchanged':True,'native600_unchanged':True,'all_control_sources_and_CPU_stage_sources_unchanged':True,'native_source_manifest_SHA256':hashlib.sha256((r/'source_manifest.json').read_bytes()).hexdigest(),'epoch':time.time()},f,indent=2)
old=r/'attention-fusion-fullmodel-control-20261028-03';oldauth=json.loads((old/'authorization.json').read_text());oldstate=json.loads((old/'final.json').read_text());assert oldstate['status']=='failed' and 'Insufficient shared budget'in oldstate['error']
for name,row in oldauth['sources'].items():assert hashlib.sha256((old/name).read_bytes()).hexdigest()==row['sha256']
for stage in oldstate['stages']:assert stage['exit_code']==0 and hashlib.sha256((old/stage['script']).read_bytes()).hexdigest()==stage['source_SHA256']
candidate=r/'attention-fusion-continuation-control-20261028-01';ca=json.loads((candidate/'authorization.json').read_text());cv=json.loads((candidate/'final.json').read_text());assert cv['status']=='failed' and 'CPU-fresh-SFT'in cv['error'];assert next(x for x in cv['stages']if x['name']=='fullstate-optimized')['exit_code']==0
for name,row in ca['sources'].items():assert hashlib.sha256((candidate/name).read_bytes()).hexdigest()==row['sha256']
for stage in cv['stages']:assert hashlib.sha256((candidate/stage['script']).read_bytes()).hexdigest()==stage['source_SHA256']
files=list(c.glob('*.json'))+list(c.glob('*.py'))+list(old.glob('*.json'))+list(old.glob('*.py'))+list(candidate.glob('*.json'))+list(candidate.glob('*.py'))+[camp/'preparation.json']
files+=list((c/'fresh-SFT-CPU-review').rglob('*.json'))
for arm in ('native','control','optimized'):
 case=r/f'runs/corex-S2048-attention-audit-{arm}-af03';assert json.loads((case/'case.json').read_text())['exit_code']==0
 files+=list(case.rglob('*.json'))
 for rank in (0,1):
  files += [camp/f'numeric/{arm}/rank-{rank}/all-state-manifest.json',camp/f'numeric/{arm}/rank-{rank}/completion.json',camp/f'semantics/{arm}/rank-{rank}/semantics.json']
for rank in (0,1):files.append(camp/f'numeric/reference/native/rank-{rank}/all-state-manifest.json')
files.append(r/'followup-control-20261027-01/full-correctness-admission.json')
files.append(r/'short-loop-CPU-control-20261026-05/primary-all-ten-run-raw-SHA.json')
files=sorted(set(files));assert all(p.suffix in ('.json','.py') and not p.is_symlink() and p.resolve().is_relative_to(r) for p in files)
assert sum(p.stat().st_size for p in files)<64*1024**2
catalog=[];buffer=io.BytesIO()
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
 for p in files:
  b=p.read_bytes();name=p.relative_to(r).as_posix();catalog.append({'name':name,'bytes':len(b),'SHA256':hashlib.sha256(b).hexdigest()});z.writestr(name,b)
 z.writestr('manifest.json',json.dumps(catalog,indent=2))
raw=buffer.getvalue();assert len(raw)<64*1024**2
with (c/'fullstate-SFT-key-export-01.zip').open('xb') as f:f.write(raw)
receipt={'ZIP_SHA256':hashlib.sha256(raw).hexdigest(),'ZIP_bytes':len(raw),'expanded_bytes':sum(x['bytes'] for x in catalog),'key_count':len(catalog),'raw_tensors_downloaded':False}
with (c/'fullstate-SFT-key-export-receipt-01.json').open('x') as f:json.dump(receipt,f,indent=2)
sys.stdout.buffer.write(json.dumps(receipt).encode()+b'\n'+raw)
'''

def read(p):return json.loads(p.read_text(encoding='utf8'))
def review(keys):
 from verify_pinning_archive import validate_coverage,WIDTHS
 c=keys/CONTROL;camp=keys/CAMP;state=read(c/'controller.json');assert state['status']=='awaiting-reviewed-stage-plan' and state['next_plan_sequence']==2
 post=read(c/'fullstate-SFT-source-postcheck.json');assert post['passed'] and post['original179_unchanged'] and post['live_original179_exact_path_derivation_unchanged'] and post['native600_unchanged'] and post['all_control_sources_and_CPU_stage_sources_unchanged']
 assert post['native_source_manifest_SHA256']=='a72ded960488d399ded2255ef55f357257e03532b8fa21a06d958170c5c98c2d'
 assert len(state['stages'])==3 and all(x['exit_code']==0 for x in state['stages'])
 auth=read(c/'authorization.json');assert auth['deadline']==1793204291 and auth['start']==1793182691
 old=keys/'attention-fusion-fullmodel-control-20261028-03';oldauth=read(old/'authorization.json')
 for name,row in oldauth['sources'].items():
  assert hashlib.sha256((old/name).read_bytes()).hexdigest()==row['sha256']==hashlib.sha256((BASE/'attention_fullmodel_20261028_v6'/name).read_bytes()).hexdigest()
 candidate=keys/'attention-fusion-continuation-control-20261028-01';ca=read(candidate/'authorization.json')
 for name,row in ca['sources'].items():assert hashlib.sha256((candidate/name).read_bytes()).hexdigest()==row['sha256']==hashlib.sha256((BASE/'attention_continuation_20261028_v1'/name).read_bytes()).hexdigest()
 local_source=BASE/'attention_CPU_recovery_20261028_v1'
 for name,row in auth['sources'].items():
  actual=hashlib.sha256((c/name).read_bytes()).hexdigest();assert actual==row['sha256']
  local=local_source/name
  if not local.exists():local=BASE/'datathinking_corex_20261027/repair_v2'/name
  assert hashlib.sha256(local.read_bytes()).hexdigest()==actual
 for stage in state['stages']:
  assert hashlib.sha256((c/stage['script']).read_bytes()).hexdigest()==stage['source_SHA256']
 for name in ('pinning_full_audit.py','verify_pinning_archive.py','verify_pinning_semantics.py','verify_sft_admission.py'):
  assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==auth['original179'][name]
 raw=read(c/'followup-incremental-raw-verification.json');assert raw['passed'] and raw['all_other_verifier_AST_identical'] and raw['only_checked_set_seed_injected'] and not raw['old_full_library_rescanned']
 assert raw['original_verifier_SHA256']==auth['original179']['verify_pinning_archive.py']
 assert raw['old_complete_receipt_SHA256']=='5ee6778f9d5b1973865d99cb43b1749cc421694bf10d1644f99756618dd00dda'
 prior=keys/'short-loop-CPU-control-20261026-05/primary-all-ten-run-raw-SHA.json';assert hashlib.sha256(prior.read_bytes()).hexdigest()==raw['old_complete_receipt_SHA256']
 assert read(prior)['passed'] and len(read(prior)['runs'])==10
 prep=read(camp/'preparation.json');seed=set();rows=[];refs={};actual_chunks=set()
 prior_rows={(x['arm'],x['rank']):x for x in read(prior)['runs']}
 for rank in (0,1):
  path=camp/f'numeric/reference/native/rank-{rank}/all-state-manifest.json';sha=hashlib.sha256(path.read_bytes()).hexdigest()
  assert sha==prior_rows['native',rank]['manifest_sha256']
  origin=next(x for x in prep['reference_manifest_and_inode_proof'] if x['rank']==rank);assert origin['manifest_SHA256']==sha
  for row in read(path)['tensors']:
   for x in row['chunks']:seed.add(('reference/'+x['file'],x['offset'],x['compressed_bytes'],x['sha256'],x['bytes']))
 for arm in ('native','control','optimized'):
  case=keys/f'runs/corex-S2048-attention-audit-{arm}-af03';assert read(case/'case.json')['exit_code']==0
  src=read(c/f'fresh-SFT-CPU-review/{arm}-source-mask.json');assert src['passed'] and src['answer_only_supervision_source_reconstructed'] and len(src['batches'])==3 and not src['cuda_initialized']
  for rank in (0,1):
   folder=camp/f'numeric/{arm}/rank-{rank}';manifest=read(folder/'all-state-manifest.json');fields=validate_coverage(manifest)
   assert len(fields)==3585 and manifest['step']==3 and manifest['passed_so_far'] and manifest['comparison_dtype']=='CPU float64' and manifest['atol']==3e-4 and manifest['rtol']==1e-3
   completion=read(folder/'completion.json');assert completion['passed'] and completion['complete_fields'] and completion['steps']==3
   if arm=='native':refs[rank]=fields
   else:
    assert fields.keys()==refs[rank].keys()
    for name,row in fields.items():assert all(row[k]==refs[rank][name][k] for k in ('shape','dtype','numel'))
   for row in fields.values():
    assert row['passed'] and sum(x['numel'] for x in row['chunks'])==row['numel']
    for x in row['chunks']:
     name=PurePosixPath(x['file']);assert not name.is_absolute() and '..' not in name.parts and '\\' not in x['file'] and x['bytes']==x['numel']*WIDTHS[row['dtype']]
     actual_chunks.add((x['file'],x['offset'],x['compressed_bytes'],x['sha256'],x['bytes']))
   entry=read(case/f'rank-{rank}/entry.json');assert entry['status']=='completed' and entry['native_files_checked']==600 and entry['completed_native_steps']==[{'step':i,'skipped':0} for i in (1,2,3)]
   policy=read(case/f'attention-policy-rank-{rank}.json');assert policy['passed'] and policy['HDO_update_calls']==3 and policy['CPU_state_layout_common_all_arms'] and policy['deterministic_algorithms'] is False
   init=policy['gradient_preallocation']['initializations'];assert len(init)==2
   for event in init:
    assert event['groups']==199 and event['elements']==3834518016 and event['buffer_created_once_before_training'] and not event['update_math_or_transfer_changed']
    for which in ('master_pages','gradient_pages'):assert event[which]['return_code']==0 and set(event[which]['sample_nodes'])=={0}
   layout=policy['CPU_state_layout'];assert layout['CPU_state_node']==0 and layout['common_all_arms'] and layout['update_formula_unchanged'] and len(layout['events'])==199
   for group in layout['events']:assert group['moment_page_query']['return_code']==0 and set(group['moment_page_query']['sample_nodes'])=={0}
   events=layout['factory_events'];assert len(events)==398
   for field in ('exp_avg','exp_avg_sq'):
    chosen=[x for x in events if x['field']==field];assert len(chosen)==199 and sum(x['elements']for x in chosen)==3834518016 and all(x['initialization_path']=='direct_local_zero'and x['copied_source_moment_bytes']==0 and x['target_node']==0 and set(x['sampled_nodes'])=={0}for x in chosen)
   attention=policy['attention'];assert attention['forward_calls']>=84 and attention['mode']==('native-softmax'if arm=='optimized'else'native') and attention['QK_baddbmm_alpha_unchanged'] and attention['BF16_output_boundary_preserved'] and attention['dropout_RNG_and_recomputation_outside_replacement']
   sha=hashlib.sha256((folder/'all-state-manifest.json').read_bytes()).hexdigest();remote_row=next(x for x in raw['runs'] if x['arm']==arm and x['rank']==rank);assert sha==remote_row['manifest_sha256']
   rows.append({'arm':arm,'rank':rank,'fields':3585,'max_abs':max(x['max_abs'] for x in fields.values()),'manifest_SHA256':sha})
 assert len(seed)==raw['reused_accepted_reference_chunks'] and len(seed|actual_chunks)==raw['unique_raw_chunks_verified'] and len(actual_chunks-seed)==raw['new_chunks_independently_decompressed_and_SHA_verified']
 sem=read(c/'followup-fresh-SFT-source-semantic-verification.json');assert sem['passed'] and not sem['cuda_initialized'] and sem['all_three_answer_source_masks_reconstructed'] and sem['tp_actual_inputs_equal'] and len(sem['runs'])==4
 assert all(x['passed'] and x['inputs_masks_scheduler_exact'] and x['rng_exact_checked'] for x in sem['runs'])
 return {'three_arm_numerical_and_SFT_semantics_passed':True,'local_independent_review_passed':True,'rows':rows,'prior_window_raw_reference_reused':True,'new_raw_chunks_independently_verified':len(actual_chunks-seed),'native_checkpoint_resume_accepted':False,'strict_long_13_loss_accepted':False,'formal_performance_accepted':False,'deadline_epoch':1793204291,'campaign':ROOT+'/'+CAMP,'source_SHA256':{n:hashlib.sha256((c/n).read_bytes()).hexdigest()for n in auth['sources']},'manifest_SHA256':{x['arm']+'/'+str(x['rank']):x['manifest_SHA256']for x in rows},'cost_policy':'coupons'}

if __name__=='__main__':
 process=subprocess.run([*SSH,'python3 -c '+shlex.quote(REMOTE)],capture_output=True,timeout=60);assert process.returncode==0,process.stderr.decode(errors='replace')[-5000:]
 header,sep,data=process.stdout.partition(b'\n');assert sep;receipt=json.loads(header);assert len(data)==receipt['ZIP_bytes']<64*1024**2 and hashlib.sha256(data).hexdigest()==receipt['ZIP_SHA256']
 bundle=AUDIT/'attention-fullstate-SFT-keys-20261028-01.zip';bundle.open('xb').write(data)
 keys=AUDIT/'attention-fullstate-SFT-keys-20261028-01';keys.mkdir()
 with zipfile.ZipFile(io.BytesIO(data)) as z:
  assert sum(x.file_size for x in z.infolist())<64*1024**2
  names=[x.filename for x in z.infolist()];assert len(names)==len(set(names))
  for name in names:
   p=PurePosixPath(name);assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name
   dest=keys.joinpath(*p.parts);assert dest.resolve().is_relative_to(keys.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.open('xb').write(z.read(name))
 catalog=read(keys/'manifest.json');assert len(catalog)==receipt['key_count'] and sum(x['bytes'] for x in catalog)==receipt['expanded_bytes']
 for x in catalog:
  b=(keys/x['name']).read_bytes();assert len(b)==x['bytes'] and hashlib.sha256(b).hexdigest()==x['SHA256']
 result=review(keys);result.update(key_receipt=receipt,both_ends_key_ZIP_SHA_verified=True,review_epoch=time.time())
 with (BASE/'results/attention-fullstate-SFT-independent-review-20261028-01.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
