"""Independently recompute final acceptance from bounded keys; never read raw tensors."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
DRIVER = BASE/'results/sft-phased-acceptance-driver-20261024-01'
DEST = BASE/'results/sft-phased-acceptance-verified-20261024-01'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def loss_rows(text):
    rows = []
    for line in text.splitlines():
        match = re.search(r'\biteration\s+(\d+)\s*/\s*120\s*\|.*?\blm loss:\s*([^\s|]+)',line)
        if match:
            number = float(match.group(2))
            if not math.isfinite(number):
                raise ValueError('Nonfinite formal loss')
            for kind in ('skipped','nan'):
                found = re.search(r'number of '+kind+r' iterations:\s*(\d+)',line)
                if not found or int(found.group(1)):
                    raise ValueError('Missing or nonzero skipped/nan counter')
            rows.append((int(match.group(1)),number))
    if [x[0] for x in rows] != [1,*range(10,121,10)]:
        raise ValueError('All13 native first/regular loss records required')
    return rows


def main():
    status = read(DRIVER/'status.json')
    if status['status']!='completed':
        raise ValueError('Wait for the unique observer final key SHA gate')
    bundle = DRIVER/'final-key-results.zip'
    checked = verify(bundle,DRIVER/'independent-final-key-verification.json',64*1024**2)
    prior = read(DRIVER/'final-key-verification.json')
    if any(checked[k]!=prior[k] for k in ('passed','bundle_sha256','verified_key_files','verified_key_bytes')):
        raise ValueError('Closed final key receipt differs')
    DEST.mkdir(exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        archive.extractall(DEST)  # verify checked every path, member, size and SHA above.
    root = DEST/'sft-phased-acceptance-20261024-01'
    controller = read(root/'controller.json')
    if controller['status']!='completed' or any(row['exit_code'] for row in controller['commands']):
        raise ValueError('Remote sequence did not close successfully')
    frozen = read(BASE/'results/sft-phased-acceptance-source-manifest-20261024-01.json')['files']
    if controller['sources']!=frozen:
        raise ValueError('Accepted source set differs')
    for name,digest in frozen.items():
        for path in (BASE/name,root/'source'/name):
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Local/actual execution source changed: '+name)
    primary = read(root/'numeric/primary-archive-verification.json')
    semantic = read(root/'semantics/semantics-verification.json')
    if (not primary['passed'] or not primary['checkpoint_resume_tested']
            or not primary['primary_complete_raw_values_verified'] or primary['unique_raw_chunks_verified']<=0
            or not semantic['passed'] or not semantic['sft_mask_gate']
            or not semantic['full_model_native_checkpoint_resume_verified']):
        raise ValueError('Fresh full values, SFT semantics and native resume required')
    if len(primary['runs'])!=10 or any(not row['passed'] for row in semantic['runs']):
        raise ValueError('All audit/restored/next-step states and semantic comparisons required')
    required = {f'step-{s}/{i}/{field}' for s in (1,2,3) for i in range(199)
                for field in ('gradient','updated_parameter','master_param','exp_avg','exp_avg_sq','step')}
    for arm in ('native','control','optimized'):
        for rank in (0,1):
            folder = root/'numeric'/arm/f'rank-{rank}'
            done = read(folder/'completion.json')
            manifest = read(folder/'all-state-manifest.json')
            fields = {x['field'] for x in manifest['tensors']}
            if (not done['passed'] or not done['complete_fields'] or done['steps']!=3
                    or {x for x in fields if '/loss/' not in x}!=required
                    or len(fields)!=len(manifest['tensors'])
                    or manifest['comparison_dtype']!='CPU float64' or manifest['atol']!=3e-4 or manifest['rtol']!=1e-3
                    or not all(x['passed'] for x in manifest['tensors'])):
                raise ValueError('Own complete199-group/three-step numeric manifest required')
    rounds = []
    for name in ('primary','confirmation'):
        if name=='confirmation' and not controller['repeat_performed']:
            break
        label = 'phased20261024'+name
        seed_means = {s:{} for s in (63100,63101,63102)}
        records = []
        for seed in seed_means:
            samples, logs = [], []
            for arm in ('native','control','optimized'):
                folder = DEST/'runs'/f'pinning-formal-{label}-{seed}-{arm}'
                case = read(folder/'case.json')
                if (case['exit_code'] or case['resource_gate_before'].strip() or case['resource_gate_after'].strip()
                        or not case['formal_performance_run'] or case['seed']!=seed or case['pinning_arm']!=arm):
                    raise ValueError('Actual fixed formal case failed')
                ranks = [read(folder/f'rank-{r}/entry.json') for r in (0,1)]
                for rank,item in enumerate(ranks):
                    if (item['rank']!=rank or item['status']!='completed' or item['completed_steps']!=120 or item['skipped_steps']
                            or item['profiler_enabled'] or item['full_state_capture_enabled'] or item['checkpoint_io_enabled']
                            or item['native_files_checked']!=600):
                        raise ValueError('Uncontaminated120-step native timer required')
                    intervals = item['raw_interval_times']
                    if ([x['end_iteration'] for x in intervals]!=list(range(10,121,10))
                            or any(x['steps']!=10 or not x['native_reset'] or not x['native_barrier']
                                   or not math.isfinite(x['seconds']) or x['seconds']<=0 for x in intervals)):
                        raise ValueError('All12 original synchronized intervals required')
                    master = item['actual_final_cpu_masters']
                    loaded = item['weight_loading']
                    if not loaded['all_source_tensors_and_native_parameters_covered'] or not loaded['all_loaded_shard_bytes_verified']:
                        raise ValueError('Actual complete official weight reconstruction required')
                    if master!={'groups':199,'elements':3834518016,'pinned_groups':199}:
                        raise ValueError('Every incremental arm must use accepted pinned masters')
                    leaves = item['actual_optimizer_state']['optimizers']
                    if len(leaves)!=1 or leaves[0]['adam_state_entries']!=199 or leaves[0]['min_step']!=120 or leaves[0]['max_step']!=120:
                        raise ValueError('Full actual AdamW update coverage required')
                retained = [max(r['raw_interval_times'][i]['seconds'] for r in ranks)/10 for i in range(2,12)]
                mean = sum(retained)/len(retained)
                seed_means[seed][arm] = mean
                records.append({'seed':seed,'arm':arm,'mean_seconds_per_step':mean,'tokens_per_second':2048/mean,
                                'retained_steps':100,'retained_intervals':retained})
                objective = [read(folder/f'sft-objective-rank-{r}.json') for r in (0,1)]
                if len(objective[0]['supervision_rows'])!=120 or objective[1]['supervision_rows']:
                    raise ValueError('Complete TP reader/broadcast SFT sample role required')
                samples.append(objective[0]['supervision_rows'])
                logs.append(loss_rows((folder/'stdout.log').read_text(encoding='utf-8')))
            if not samples[0]==samples[1]==samples[2]:
                raise ValueError('Paired actual SFT samples/masks differ')
            for values in logs[1:]:
                if any(abs(x[1]-y[1])>3e-4+1e-3*abs(y[1]) for x,y in zip(values,logs[0])):
                    raise ValueError('Paired formal logged losses differ beyond unchanged tolerance')
        ratios = {arm:[seed_means[s][arm]/seed_means[s]['optimized'] for s in seed_means] for arm in ('native','control')}
        geomeans = {arm:math.exp(sum(math.log(x) for x in values)/3) for arm,values in ratios.items()}
        cvs = {arm:statistics.stdev(seed_means[s][arm] for s in seed_means)/statistics.mean(seed_means[s][arm] for s in seed_means)
               for arm in ('native','control','optimized')}
        promoted = all(geomeans[arm]>1.01 and all(x>1 for x in ratios[arm]) for arm in ratios)
        repeat = any(cvs[arm]>=.1 for arm in ('native','control'))
        remote = read(root/('formal-'+name+'.json'))
        if remote['round_promotion_passed']!=promoted or remote['repeat_required']!=repeat:
            raise ValueError('Independently recomputed promotion/CV gate differs')
        for arm,key in (('native','vs_native'),('control','vs_control')):
            if abs(remote[key]['geomean_throughput_ratio']-geomeans[arm])>1e-12:
                raise ValueError('Independent paired ratio differs')
        if name=='primary' and repeat and not controller['repeat_performed']:
            raise ValueError('Original CV rule requires a complete independent confirmation')
        rounds.append({'name':name,'cases':records,'ratios':ratios,'geomeans':geomeans,'seed_mean_cv':cvs,
                       'promotion_passed':promoted,'repeat_required':repeat})
    promoted = all(row['promotion_passed'] for row in rounds)
    if promoted!=controller['formal_performance_passed'] or promoted!=status['formal_performance_passed']:
        raise ValueError('Final candidate decision differs')
    result = {'schema':'jiuge7b-phased-acceptance-independent-review-v1','passed':True,
              'formal_performance_passed':promoted,'incremental_baseline':'accepted pinned repair,unbound',
              'comparison_control':'same pinning,two-phase affinity,native moments','rounds':rounds,
              'numeric_raw_verified_at_ACT':primary['unique_raw_chunks_verified'],
              'fresh_SFT_state_semantics_and_native_resume_passed':True,'raw_tensors_downloaded':False,
              'old_CPT84_4_or_SFT92_9_or_screen9_2_reused':False,'key_bundle_sha256':checked['bundle_sha256']}
    output = BASE/'results/sft-phased-acceptance-independent-review-20261024-01.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps({'passed':True,'formal_performance_passed':promoted,'rounds':len(rounds)}),flush=True)


if __name__=='__main__':
    main()
