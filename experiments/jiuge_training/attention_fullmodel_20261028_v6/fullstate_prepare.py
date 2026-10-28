"""Fresh complete numerical/semantic campaign, preserving old accepted raw."""
import hashlib,json,os,sys,time
from pathlib import Path
r=Path(os.environ['JIUGE_DT_ROOT']);c=Path(os.environ['JIUGE_DT_CONTROL']);sys.path[:0]=[str(c),str(r)]
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and os.getuid()==1001 and time.time()<1793183838-1800
admission=json.loads((c/'component-independent-admission.json').read_text());assert admission['component_accepted'] and admission['closed_cases']==24 and admission['all_output_QKV_SHA_equal'] and admission['atol']==3e-4 and admission['rtol']==1e-3
for name in('attention_fusion_adapter.py','fused_scale_mask.py','fused_scale_mask_softmax.py','corex_process_guard_data.py'):assert hashlib.sha256((c/name).read_bytes()).hexdigest()==admission['source_SHA256'][name]

for mode in ('default','strict'):
 for rank in (0,1):
  q=r/f'runs/corex-S2048-fixed-model-repeat-{mode}-rp01/fixed-model-repeat-rank-{rank}/result.json';v=json.loads(q.read_text())
  assert v['completed'] and v['repeat_calls']==16 and v['real_optimizer_updates']==0 and v['weights_unchanged_verified']
  assert v['passed_repeatability']==(mode=='strict')
old=r/'numa-loss-repair-control-20261027-06';closed=json.loads((old/'final.json').read_text());assert closed['status']=='failed' and not Path('/proc/'+str(closed['PID'])).exists()
assert 'No running processes found' in closed['GPU_final']['selected_pair']
for row in json.loads((r/'followup-control-20261027-01/full-correctness-admission.json').read_text())['verified_receipts']:
 p=Path(row['file']);assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'] and json.loads(p.read_text())['passed']
with (c/'numa-plan.json').open('xb') as f:f.write((old/'numa-plan.json').read_bytes())
camp=r/'CoreX-attention-fusion-20261028-03';assert not camp.exists();camp.mkdir()
from pinning_candidate import prepare
prepare(camp/'numeric');(camp/'checkpoints').mkdir();(camp/'semantics').mkdir()
reference=r/'CoreX-correctness-20261026-02/numeric';assert reference.is_dir()
(camp/'numeric/reference').mkdir()
manifest=[]
for rank in (0,1):
 q=reference/'native'/f'rank-{rank}'/'all-state-manifest.json';b=q.read_bytes();v=json.loads(b)
 assert v['step']==3 and v['passed_so_far'] and len(v['tensors'])==3585
 target_manifest=camp/'numeric/reference/native'/f'rank-{rank}'/'all-state-manifest.json';target_manifest.parent.mkdir(parents=True,exist_ok=True)
 with target_manifest.open('xb') as f:f.write(b)
 assert hashlib.sha256(target_manifest.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()
 chunks={x['file'] for row in v['tensors'] for x in row['chunks']}
 rows=[]
 for name in chunks:
  source=(reference/name).resolve(strict=True);assert source.is_relative_to(reference.resolve()) and source.suffix=='.zlibpack' and source.stat().st_uid==1001
  view=camp/'numeric/reference'/name;assert view.resolve().is_relative_to((camp/'numeric').resolve());view.parent.mkdir(parents=True,exist_ok=True);a=source.stat();os.link(source,view);bstat=view.stat();assert not view.is_symlink();assert (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)==(bstat.st_dev,bstat.st_ino,bstat.st_size,bstat.st_mtime_ns)
  rows.append({'source':str(source),'view':str(view),'device':a.st_dev,'inode':a.st_ino,'bytes':a.st_size,'mtime_ns':a.st_mtime_ns,'source_ctime_before_link_ns':a.st_ctime_ns,'source_ctime_after_link_ns':source.stat().st_ctime_ns,'source_nlink_before':a.st_nlink,'source_nlink_after':source.stat().st_nlink,'hardlink_payload_not_modified':True})
 manifest.append({'rank':rank,'manifest_SHA256':hashlib.sha256(b).hexdigest(),'same_inode_reference_files':rows})
# Exercise the unchanged private-cache guard with the exact alias layout on tiny owned fixtures.
from corex_own_audit_cache import OwnCoreXAuditCache
fixture=c/'cache-scope-CPU-preflight';fixture.mkdir();(fixture/'numeric').mkdir();(fixture/'checkpoint').mkdir()
private=fixture/'outside-private.zlibpack';private.write_bytes(b'own-cache-scope-fixture');hard=fixture/'numeric/inside.zlibpack';os.link(private,hard)
cache=OwnCoreXAuditCache(fixture/'numeric',fixture/'checkpoint');cache.advise(hard)
assert private.read_bytes()==b'own-cache-scope-fixture' and hard.stat().st_ino==private.stat().st_ino
leaf=fixture/'numeric/symlink.zlibpack';leaf.symlink_to(private)
parent=fixture/'numeric/parent-link';parent.symlink_to(fixture,target_is_directory=True)
wrong=fixture/'numeric/wrong.txt';wrong.write_bytes(b'private')
refused=[]
for path in (private,leaf,parent/'outside-private.zlibpack',wrong):
 try:cache.advise(path)
 except ValueError:refused.append(str(path))
 else:raise AssertionError('Original cache-scope negative was not refused')
assert len(refused)==4
# Untimed advisory reclaim of only the new campaign's owned same-inode reference files.
# Keep MemFree admission and all live allocation guards unchanged.
reclaim_before={p.parent.name:p.read_text() for p in Path('/sys/devices/system/node').glob('node[01]/meminfo')}
reference_cache=OwnCoreXAuditCache(camp/'numeric',camp/'checkpoints')
for row in manifest:
 for item in row['same_inode_reference_files']:
  view=Path(item['view']);a=view.stat();assert a.st_uid==1001 and a.st_ino==item['inode'] and a.st_size==item['bytes'] and a.st_mtime_ns==item['mtime_ns'];reference_cache.advise(view)
reclaim_after={p.parent.name:p.read_text() for p in Path('/sys/devices/system/node').glob('node[01]/meminfo')}
source=r/'pinning_full_audit.py';frozen=json.loads((c/'authorization.json').read_text())['original179']['pinning_full_audit.py'];assert hashlib.sha256(source.read_bytes()).hexdigest()==frozen
result={'prepared':True,'diagnostic_only_admitted':True,'full_correctness_or_long_loss_or_performance_accepted':False,'reference_layout':'real directories + owned same-inode hardlinked raw leaves; copied SHA-bound manifests','original_cache_guard_unchanged':True,'original_cache_scope_CPU_positive_passed':True,'original_cache_scope_CPU_negative_refused':refused,'scope':'fresh complete three-arm strict numerical/semantic validation, no performance admission','native_original_reference_bound_extra_comparison':True,'original_CPU_float64_comparator_SHA256':frozen,'atol':3e-4,'rtol':1e-3,'reference_manifest_and_inode_proof':manifest,'large_prior_SHA_rescanned':False,'untimed_private_reference_cache_advice':{'calls':reference_cache.calls,'before':reclaim_before,'after':reclaim_after,'payload_modified':False,'global_drop_caches':False},'new_correctness_accepted':False,'deadline_epoch':1793183838}
with (camp/'preparation.json').open('x') as f:json.dump(result,f,indent=2)
with (c/'full-model-diagnostic-admission.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result),flush=True)
