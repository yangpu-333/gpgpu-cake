"""Run complete remote affinity gates; keep only bounded keys on Windows."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import zipfile
from analyze_pinning_formal import summarize,evaluate,SEEDS,ARMS
from pinning_key_results import verify

BASE=Path(__file__).resolve().parent
PREP='/home/huids25/kda-jiuge-prep-20261014'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--host',default='192.168.5.194')
    parser.add_argument('--known-hosts',type=Path,required=True)
    args=parser.parse_args()
    folder=BASE/'results/cpu-numa-driver-20261016-01'; folder.mkdir(exist_ok=False)
    shutil.copyfile(__file__,folder/Path(__file__).name)
    options=['-i',str(Path.home()/'.ssh/id_ed25519'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
             '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=30',
             '-o','ServerAliveCountMax=6','-o','UserKnownHostsFile='+str(args.known_hosts.resolve())]
    host='huids25@'+args.host; ssh=['ssh',*options,'-p',str(args.port),host]; scp=['scp',*options,'-P',str(args.port)]
    state={'schema':'jiuge7b-cpu-numa-local-driver-v1','status':'checking_remote',
           'started_utc':datetime.now(timezone.utc).isoformat(),'raw_original_download_enabled':False,
           'remote_probes_inside_formal':False,'formal_performance_passed':False,
           'host':args.host,'port':args.port,'expected_hostname':'sg-huids25-261016-13441-default0-0'}
    def save():
        state['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=folder/'status.tmp'; temp.write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8'); temp.replace(folder/'status.json')
    def remote(command,timeout=60):
        value=subprocess.run([*ssh,command],capture_output=True,text=True,timeout=timeout)
        if value.returncode:
            raise RuntimeError('ACT command failed: '+value.stderr[-4000:])
        return value.stdout
    def stream(name,command,timeout):
        state['status']=name; save()
        with (folder/(name+'-stdout.log')).open('x') as stdout,(folder/(name+'-stderr.log')).open('x') as stderr:
            value=subprocess.run([*ssh,command],stdout=stdout,stderr=stderr,timeout=timeout)
        if value.returncode:
            raise RuntimeError(name+' stopped; preserve remote originals and local console logs')
    def fetch(name):
        source=PREP+'/'+name+'.zip'; target=folder/(name+'.zip')
        size=int(remote('stat -c %s '+source).strip())
        if not 0<size<=68*1024**2:
            raise ValueError('Remote key ZIP exceeds local budget')
        subprocess.run([*scp,host+':'+source,str(target)],check=True,timeout=600)
        if target.stat().st_size!=size:
            raise ValueError('Incomplete key transfer')
        proof_path=folder/(name+'-verification.json'); proof=verify(target,proof_path)
        return target,proof_path,proof
    def zip_objects(path):
        with zipfile.ZipFile(path) as archive:
            return {name:json.loads(archive.read(name)) for name in archive.namelist() if name.endswith('.json')}
    save()
    try:
        if remote('hostname').strip()!=state['expected_hostname']:
            raise ValueError('Wrong allocated host')
        stream('running_full_validation','cd '+PREP+' && .venv/bin/python -u continue_numa_validation.py',21000)
        validation_root=PREP+'/cpu-numa-validation-20261016-01'
        control=json.loads(remote('cat '+validation_root+'/controller.json'))
        (folder/'validation-controller-final.json').write_text(json.dumps(control,indent=2)+'\n',encoding='utf-8')
        if control['status']!='ready_for_local_key_verification':
            raise ValueError('Complete numerical/restore/archive gate not ready')
        for name,digest in control['source_sha256'].items():
            if hashlib.sha256((BASE/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Local expected source differs from actual validation: '+name)
        remote('cd '+PREP+' && .venv/bin/python curate_cpu_numa_keys.py --stage validation',timeout=300)
        bundle,receipt_path,proof=fetch('cpu-numa-curated-validation-key-results-20261016')
        objects=zip_objects(bundle)
        def key(suffix):
            values=[value for name,value in objects.items() if name.endswith(suffix)]
            if len(values)!=1:
                raise ValueError('One actual key object required: '+suffix)
            return values[0]
        raw=key('/cpu-numa-validation-20261016-01/full-archive-verification.json')
        semantics=key('/cpu-numa-validation-20261016-01/semantics-verification.json')
        if not raw['passed'] or not raw['primary_complete_raw_values_verified'] or not semantics['passed'] or not semantics['tp_actual_inputs_equal']:
            raise ValueError('Concrete ACT numerical/semantic proofs did not pass')
        expected={'optimized':3585,'resume-restored':995,'resume-next-step':1195}
        rows=raw['new_candidate_and_resume_runs']
        if {(r['arm'],r['rank']) for r in rows}!={(arm,rank) for arm in expected for rank in (0,1)}:
            raise ValueError('Complete new three-step and resumed state comparisons required')
        if any(row['fields']!=expected[row['arm']] for row in rows):
            raise ValueError('Actual field coverage differs')
        for arm in expected:
            for rank in (0,1):
                manifest=key(f'/cpu-numa-validation-20261016-01/{arm}/rank-{rank}/all-state-manifest.json')
                if manifest['comparison_dtype']!='CPU float64' or manifest['atol']!=3e-4 or manifest['rtol']!=1e-3 or not all(row['passed'] for row in manifest['tensors']):
                    raise ValueError('Original numerical comparison standard not preserved')
        review={'passed':True,'actual_full_state_and_restore_rows':rows,'semantic_passed':semantics['passed'],
                'primary_raw_chunks_verified':raw['unique_raw_chunks_verified'],'local_key_verification':proof,
                'formal_performance_passed':False}
        (folder/'independent-validation-review.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf-8')
        subprocess.run([*scp,str(receipt_path),host+':'+validation_root+'/local-key-verification.json'],check=True,timeout=60)
        admission=validation_root+'/formal-admission.json'
        admission_json=json.loads(remote('cd '+PREP+' && .venv/bin/python numa_admission.py --root '+validation_root+
                ' --local-key-receipt '+validation_root+'/local-key-verification.json --output '+admission))
        (folder/'formal-admission.json').write_text(json.dumps(admission_json,indent=2)+'\n',encoding='utf-8')
        # From here only the original SSH console is open; no ACT polling or SCP
        # until every required primary/confirmation timer interval has closed.
        stream('running_formal','cd '+PREP+' && .venv/bin/python -u continue_numa_formal.py --admission '+admission,21000)
        formal_root=PREP+'/cpu-numa-formal-20261016-01'
        formal=json.loads(remote('cat '+formal_root+'/controller.json'))
        (folder/'formal-controller-final.json').write_text(json.dumps(formal,indent=2)+'\n',encoding='utf-8')
        if formal['status'] not in ('completed_promoted','completed_not_promoted'):
            raise ValueError('Formal benchmark incomplete; no promotion')
        command='cd '+PREP+' && .venv/bin/python pinning_key_results.py --root '+PREP+' --folder '+formal_root
        for case in formal['completed_cases']:
            if not case.replace('-','').isalnum():
                raise ValueError('Unexpected case identity')
            command+=' --folder '+PREP+'/runs/'+case
        command+=' --output '+PREP+'/cpu-numa-formal-key-results-20261016.zip'
        remote(command,timeout=300)
        formal_bundle,formal_receipt,formal_proof=fetch('cpu-numa-formal-key-results-20261016')
        actual=zip_objects(formal_bundle); independent=[]
        for round_record in formal['rounds']:
            label=round_record['round']; means={seed:{} for seed in SEEDS}; summaries=[]
            for seed in SEEDS:
                for arm in ARMS:
                    case=f'cpu-numa-formal-{label}-{seed}-{arm}'
                    case_record=actual[f'runs/{case}/case.json']
                    if (case_record['exit_code'] or case_record['resource_gate_before'].strip() or case_record['resource_gate_after'].strip()
                        or case_record['numa_arm']!=arm or case_record['environment']['OMP_NUM_THREADS']!='8'):
                        raise ValueError('Independent review found failed/contended case')
                    ranks=[actual[f'runs/{case}/rank-{rank}/entry.json'] for rank in (0,1)]
                    if any(r['actual_final_cpu_masters']!={'groups':199,'elements':3834518016,'pinned_groups':199} for r in ranks):
                        raise ValueError('All fresh arms must retain pinned masters')
                    for record in ranks:
                        loaded=record['weight_loading']; optimizers=record['actual_optimizer_state']['optimizers']
                        if not loaded['all_source_tensors_and_native_parameters_covered'] or not loaded['all_loaded_shard_bytes_verified']:
                            raise ValueError('Actual original weight byte verification incomplete')
                        if len(optimizers)!=1 or optimizers[0]['adam_state_entries']!=199 or optimizers[0]['min_step']!=120 or optimizers[0]['max_step']!=120:
                            raise ValueError('All actual optimizer states must reach step120')
                        if arm!='native':
                            binding=actual[f"runs/{case}/numa-binding-rank-{record['rank']}.json"]
                            if binding['mode']!=('local8' if arm=='optimized' else 'unbound'):
                                raise ValueError('Independent review found wrong affinity dispatch')
                            if any(admission_json['frozen_source_sha256'][name]!=digest for name,digest in binding['source_sha256'].items()):
                                raise ValueError('Formal affinity source differs from full-state admission')
                    summary=summarize(ranks); means[seed][arm]=summary['mean_seconds_per_step']; summaries.append({'case':case,**summary})
            evaluated=evaluate(means)
            for comparison in ('vs_native','vs_control'):
                if abs(evaluated[comparison]['geomean_throughput_ratio']-round_record[comparison]['geomean_throughput_ratio'])>1e-12:
                    raise ValueError('Local independent formal statistics differ')
            evaluated['arm_aggregate']={arm:{'mean_seconds_per_step':statistics.mean(means[seed][arm] for seed in SEEDS),
                  'effective_tokens_per_second':2048/statistics.mean(means[seed][arm] for seed in SEEDS)} for arm in ARMS}
            independent.append({'round':label,'raw_case_summaries':summaries,**evaluated})
        (folder/'independent-formal-review.json').write_text(json.dumps({'rounds':independent,'local_key_verification':formal_proof,
            'formal_performance_passed':formal['formal_performance_passed']},indent=2)+'\n',encoding='utf-8')
        state.update(status=formal['status'],formal_performance_passed=formal['formal_performance_passed'],
                     rounds=[{key:value for key,value in row.items() if key!='raw_case_summaries'} for row in independent],
                     formal_key_bundle_sha256=formal_proof['bundle_sha256'],validation_key_bundle_sha256=proof['bundle_sha256'])
    except BaseException as exc:
        state.update(status='stopped',error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        save(); print(json.dumps(state),flush=True)


if __name__=='__main__':
    main()
