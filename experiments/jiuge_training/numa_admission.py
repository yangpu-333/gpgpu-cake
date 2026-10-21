"""Concrete full-state, restore and bounded local-key admission for affinity."""
import hashlib
import json
from pathlib import Path

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
PINNED_SHA='8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88'


def check(path):
    value=json.loads(Path(path).read_text())
    if value.get('schema')!='jiuge7b-cpu-numa-formal-admission-v1' or not value.get('passed'):
        raise ValueError('New affinity admission required')
    if value['pinned_candidate_sha256']!=PINNED_SHA:
        raise ValueError('Pinned optimizer changed')
    expected={'jiuge7b-cpu-numa-full-archive-verification-v1','jiuge7b-cpu-numa-semantics-verification-v1',
              'jiuge-pinning-local-key-verification-v1'}
    schemas=set()
    for row in value['proofs']:
        data=Path(row['file']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256']:
            raise ValueError('Concrete admission proof changed')
        proof=json.loads(data)
        if not proof.get('passed'):
            raise ValueError('Underlying proof did not pass')
        schemas.add(proof['schema'])
    if len(value['proofs'])!=3 or schemas!=expected:
        raise ValueError('Complete new candidate raw/semantic/local gates required')
    for name,digest in value['frozen_source_sha256'].items():
        if hashlib.sha256((PREP/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Affinity source changed after correctness admission: '+name)
    return value


def create(root,local_key_receipt,output):
    controller=json.loads((root/'controller.json').read_text())
    if controller['status']!='ready_for_local_key_verification':
        raise ValueError('Complete new numerical/resume/raw audit first')
    proof_paths=[root/'full-archive-verification.json',root/'semantics-verification.json',local_key_receipt]
    value={'schema':'jiuge7b-cpu-numa-formal-admission-v1','passed':True,
           'pinned_candidate_sha256':PINNED_SHA,'validation_root':str(root),
           'frozen_source_sha256':controller['source_sha256'],
           'proofs':[{'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for path in proof_paths],
           'incremental_baseline':'Accepted pinned optimizer with unbound affinity; new all-arm timings required',
           'formal_performance_passed':False}
    with output.open('x') as stream:
        json.dump(value,stream,indent=2); stream.write('\n')
    return check(output)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--local-key-receipt',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(create(args.root,args.local_key_receipt,args.output),indent=2))
