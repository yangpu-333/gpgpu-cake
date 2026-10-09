"""Seal final stage34 decision after complete local original-byte verification."""
import json
from pathlib import Path
import pstats
import verify_receipts as v

HERE=Path(__file__).resolve().parent
BASE=HERE/'evidence'


def write_new(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    decision_path=BASE/'decision.json'
    v.require(not decision_path.exists(),'immutable decision already exists')
    receipt=BASE/'raw-backup-moe-final-receipt.json'
    r=v.read(receipt)
    v.require(r['status']=='verified' and r['through_candidate']=='004'
              and r['gpu_timing_finished_acknowledged'] is True and r['remote_deleted'] is False,
              'complete final original backup required')
    v.require(v.sha(BASE/'raw-inventory-moe-final.json')==r['inventory_sha256']
              and r['backup_script_sha256']==v.sha(HERE/'backup_raw.py'), 'backup provenance differs')
    for row in r['files']:
        path=BASE/'raw-backup'/row['path']
        v.require(path.stat().st_size==row['bytes'] and v.sha(path)==row['sha256'], 'backup raw bytes differ')
    records={}
    for version in ('002','003','004'):
        proof=v.verify(version,BASE,True)
        proof.update(candidate_sha256=v.sha(HERE/'candidates'/version/'candidate.py'),
                     performance_sha256=v.sha(BASE/('performance-'+version+'.json')),
                     complete_backup_receipt_sha256=v.sha(receipt))
        output=BASE/('cpu-verify-'+version+'-raw.json');write_new(output,proof)
        records[version]={'candidate_sha256':proof['candidate_sha256'],
            'proof_sha256':v.sha(output),'performance_sha256':proof['performance_sha256'],
            'comparisons':proof['comparisons'],'vs_control':proof['vs_control'],
            'vs_native':proof['vs_native'],'promoted':proof['promoted']}
    confirmed=v.verify('004',BASE,True,True)
    confirmed.update(candidate_sha256=v.sha(HERE/'candidates/004/candidate.py'),
                     performance_sha256=v.sha(BASE/'confirmation/performance-004.json'),
                     complete_backup_receipt_sha256=v.sha(receipt))
    proof_path=BASE/'cpu-verify-004-confirmation-raw.json';write_new(proof_path,confirmed)
    records['004-confirmation']={'candidate_sha256':confirmed['candidate_sha256'],
        'proof_sha256':v.sha(proof_path),'performance_sha256':confirmed['performance_sha256'],
        'vs_control':confirmed['vs_control'],'vs_native':confirmed['vs_native'],'promoted':confirmed['promoted']}
    profiles={}
    originals=v.read(BASE/'raw-host-profiles.json')['files']+v.read(BASE/'raw-host-profile004.json')['files']
    for row in originals:
        path=HERE/row['path']
        v.require(path.stat().st_size==row['bytes'] and v.sha(path)==row['sha256'],'original host profile differs')
        stats=pstats.Stats(str(path))
        calls=sum(value[1] for (file,line,name),value in stats.stats.items()
                  if file.endswith('/triton/compiler/compiler.py') and name=='compile')
        profiles[path.parent.name]={'original_sha256':row['sha256'],'triton_compile_calls':calls}
    measured=v.read(BASE/'confirmation/performance-004.json')
    final_promoted=records['004']['promoted'] and records['004-confirmation']['promoted']
    decision={'schema':'moe-weighted-stage34-final-decision-v1','date':'2026-10-09',
        'status':'complete','selected_candidate':'004' if final_promoted else None,
        'promoted':final_promoted,'decision':'promoted_tested_model_only' if final_promoted else 'not_promoted_after_repeat',
        'rounds':records,'failed_candidate001':{'reason':'two step3 embedding gradients outside unchanged tolerance',
            'candidate_sha256':v.sha(HERE/'candidates/001/candidate.py'),
            'comparison_sha256':v.sha(BASE/'comparison-native-001.json'),
            'diagnostic_sha256':v.sha(BASE/'diagnostics-001.json')},
        'baseline':'same-round native pinned SequentialMLP with compiled weighted SwiGLU/default fused causal softmax; common explicit uint8 stable-sort compatibility',
        'control':'native + frozen CE0008 + gradnorm001; no Dense SwiGLU004/Attention003/RMSNorm candidate',
        'optimized':'control + MoE weighted SwiGLU004 only',
        'contract_sha256':v.sha(HERE/'contract.json'),'model':measured['model'],
        'performance_sha256':v.sha(BASE/'confirmation/performance-004.json'),
        'first_round_performance_sha256':v.sha(BASE/'performance-004.json'),
        'promotion_requires_both_independent_rounds':True,
        'primary_metric':measured['metric'],'profile_counts_diagnostic_only':profiles,
        'raw_backup':{'receipt_sha256':v.sha(receipt),'files':r['verified_files'],
            'raw_bytes':r['raw_bytes'],'unique_files':r['unique_files'],'unique_bytes':r['unique_bytes'],
            'remote_deleted':False},
        'scope':measured['scope'],'cpu_arithmetic_replay':False,
        'scope_limits':['three-update numerical gates are separate from120-step timing',
            'single GPU mock data, dropout0, SGD; no production/convergence/AdamW/multiGPU/GroupedGEMM/SFT/RL claim'],
        'sealer_sha256':v.sha(__file__),'cpu_verifier_sha256':v.sha(HERE/'verify_receipts.py')}
    write_new(decision_path,decision)
    print(json.dumps({k:decision[k] for k in ('status','decision','selected_candidate','promoted','raw_backup','profile_counts_diagnostic_only')}))


if __name__=='__main__':main()
