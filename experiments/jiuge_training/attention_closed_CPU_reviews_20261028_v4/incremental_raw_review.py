"""Frozen six-run verifier, seeded only with accepted, unchanged raw references.

New chunks still take its original decompression/length/SHA path. Old raw is
not rescanned; its complete accepted receipt, manifests and inode identity bind it.
"""
import ast,copy,hashlib,json,os,sys,time
from pathlib import Path

def identity(path,row):
    stat=path.stat()
    assert not path.is_symlink()
    assert (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)==(row['device'],row['inode'],row['bytes'],row['mtime_ns'])
    assert stat.st_uid==1001

def verifier_with_seed(body,namespace,seed):
    tree=ast.parse(body);fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='verify')
    before=copy.deepcopy(fn);matches=[]
    for i,x in enumerate(fn.body):
        if isinstance(x,ast.Assign) and len(x.targets)==1 and isinstance(x.targets[0],ast.Name) and x.targets[0].id=='checked':matches.append(i)
    assert len(matches)==1
    i=matches[0];assert ast.dump(fn.body[i].value,include_attributes=False)==ast.dump(ast.parse('set()',mode='eval').body,include_attributes=False)
    fn.body[i].value=ast.parse('set(_accepted_reference_keys)',mode='eval').body
    restored=copy.deepcopy(fn);restored.body[i]=before.body[i]
    assert ast.dump(restored,include_attributes=False)==ast.dump(before,include_attributes=False)
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    ns=dict(namespace);ns['_accepted_reference_keys']=seed
    exec(compile(module,'<original-archive-verifier-with-proven-reference-seed>','exec'),ns)
    return ns['verify'],hashlib.sha256(ast.dump(before,include_attributes=False).encode()).hexdigest()

def main():
    root=Path(os.environ['JIUGE_DT_ROOT']);control=Path(os.environ['JIUGE_DT_CONTROL'])
    assert os.getuid()==1001 and os.environ['CUDA_VISIBLE_DEVICES']=='' and time.time()<1793204291-180
    sys.path[:0]=[str(control),str(root)]
    from corex_process_guard_data import enforce_rank_limit
    enforce_rank_limit()
    import verify_pinning_archive as frozen
    original=root/'verify_pinning_archive.py';body=original.read_bytes()
    expected=json.loads((control/'authorization.json').read_text())['original179']['verify_pinning_archive.py']
    assert hashlib.sha256(body).hexdigest()==expected and Path(frozen.__file__).resolve()==original.resolve()
    admission=json.loads((root/'followup-control-20261027-01/full-correctness-admission.json').read_text())
    assert admission['passed'] and admission['primary_complete_raw_values_verified']
    accepted={x['file']:x['sha256'] for x in admission['verified_receipts']}
    prior=root/'short-loop-CPU-control-20261026-05/primary-all-ten-run-raw-SHA.json'
    assert hashlib.sha256(prior.read_bytes()).hexdigest()==accepted[str(prior)]
    raw=json.loads(prior.read_text());assert raw['passed'] and raw['primary_complete_raw_values_verified'] and raw['checkpoint_resume_tested'] and len(raw['runs'])==10
    oldroot=root/'CoreX-correctness-20261026-02/numeric'
    numeric=root/'CoreX-attention-fusion-20261028-03/numeric'
    prep=json.loads((numeric.parent/'preparation.json').read_text())
    assert prep['prepared'] and not prep['large_prior_SHA_rescanned']
    oldruns={(x['arm'],x['rank']):x for x in raw['runs']}
    seeded=set();packs={};manifest_proof=[]
    for rank in (0,1):
        info=next(x for x in prep['reference_manifest_and_inode_proof'] if x['rank']==rank)
        manifest=numeric/f'reference/native/rank-{rank}/all-state-manifest.json'
        expected_manifest=oldruns['native',rank]['manifest_sha256']
        assert info['manifest_SHA256']==hashlib.sha256(manifest.read_bytes()).hexdigest()==expected_manifest
        assert hashlib.sha256((oldroot/f'native/rank-{rank}/all-state-manifest.json').read_bytes()).hexdigest()==expected_manifest
        byview={x['view']:x for x in info['same_inode_reference_files']}
        value=json.loads(manifest.read_text());frozen.validate_coverage(value)
        for row in value['tensors']:
            assert row['passed']
            for chunk in row['chunks']:
                path=numeric/'reference'/chunk['file'];assert path.resolve(strict=True).is_relative_to(numeric.resolve())
                origin=byview[str(path)];identity(path,origin);identity(Path(origin['source']),origin)
                assert Path(origin['source']).resolve()==(oldroot/chunk['file']).resolve()
                key=(str(path.relative_to(numeric)),chunk['offset'],chunk['compressed_bytes'],chunk['sha256'],chunk['bytes'])
                seeded.add(key);packs[str(path)]=origin
        manifest_proof.append({'rank':rank,'old_accepted_manifest_SHA256':expected_manifest,'new_reference_manifest_SHA256':hashlib.sha256(manifest.read_bytes()).hexdigest()})
    run,ast_sha=verifier_with_seed(body,vars(frozen),seeded)
    result=run(numeric,False)
    for path,origin in packs.items():identity(Path(path),origin);identity(Path(origin['source']),origin)
    assert result['passed'] and len(result['runs'])==6 and not result['checkpoint_resume_tested']
    result.update(storage_policy='Complete originals remain Data Thinking; bounded local keys only',
        original_verifier_SHA256=expected,original_verifier_function_AST_SHA256=ast_sha,
        only_checked_set_seed_injected=True,all_other_verifier_AST_identical=True,
        old_complete_receipt_SHA256=hashlib.sha256(prior.read_bytes()).hexdigest(),
        old_reference_manifest_proof=manifest_proof,unchanged_same_inode_packs=list(packs.values()),
        reused_accepted_reference_chunks=len(seeded),new_chunks_independently_decompressed_and_SHA_verified=result['unique_raw_chunks_verified']-len(seeded),
        prior_window_raw_reference_reused=True,old_full_library_rescanned=False,
        native_checkpoint_resume_accepted=False,formal_performance_passed=False,deadline_epoch=1793204291)
    with (control/'followup-incremental-raw-verification.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
