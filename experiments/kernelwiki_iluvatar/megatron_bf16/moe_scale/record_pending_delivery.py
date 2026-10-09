"""Record GPU-complete receipts without claiming the interrupted raw backup is complete."""
import json
from pathlib import Path
import verify_receipts as v

HERE = Path(__file__).resolve().parent


def main():
    output = HERE/'evidence/delivery-pending-backup.json'
    v.require(not output.exists(), 'immutable pending delivery receipt already exists')
    v.require(not (HERE/'evidence/decision.json').exists(), 'final decision already exists')
    models = {model:v.verify(model) for model in ('long','experts')}
    inventory = v.read(HERE/'evidence/raw-inventory-moe-final.json')
    objects = HERE/'evidence/raw-backup/objects'
    expected = {row['sha256']:row['bytes'] for row in inventory}
    verified = []
    for digest, size in expected.items():
        path = objects/digest
        if path.exists():
            v.require(path.stat().st_size==size and v.sha(path)==digest, 'partial-backup verified object differs')
            verified.append({'sha256':digest,'bytes':size})
    receipt = {'schema':'moe-heldout-stage35-delivery-pending-v1','date':'2026-10-09',
        'status':'gpu_complete_raw_backup_pending','models':models,
        'reader_sha256':v.sha(HERE/'verify_receipts.py'),
        'controllers_manifest_sha256':v.sha(HERE/'models/controllers.json'),
        'raw_inventory_sha256':v.sha(HERE/'evidence/raw-inventory-moe-final.json'),
        'raw_files':len(inventory),'raw_bytes':sum(row['bytes'] for row in inventory),
        'unique_files':len(expected),'unique_bytes':sum(expected.values()),
        'verified_downloaded_objects':verified,
        'remote_originals':'Pod terminal confirms /tmp/kda-moe-scale-stage35 exists; SSH ingress currently fails before banner.',
        'backup_complete':False,'final_skill_publication_complete':False,
        'released_skill_stage':34,'final_decision_written':False,
        'failure':'SSH ic.h3i.buaa.edu.cn:41686 refused; local TCP probe also failed. Remote SSH listener exists per user terminal.',
        'scope':'Two fixed GPU model experiments are complete; this immutable receipt precedes complete original-byte backup and stage35 skill publication. CPU does not rerun GPU arithmetic.'}
    with output.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':receipt['status'],'raw_files':receipt['raw_files'],
        'verified_objects':len(verified),'verified_bytes':sum(row['bytes'] for row in verified),
        'models':{name:{key:proof[key] for key in ('promoted','vs_control','vs_native','comparisons')}
                  for name,proof in models.items()}}))


if __name__=='__main__':
    main()
