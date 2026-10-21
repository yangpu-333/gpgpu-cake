"""Independently recompute closed SFT formal results from verified local keys."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

from analyze_pinning_formal import analyze
from pinning_key_results import verify

BASE = Path(__file__).resolve().parent
DRIVER = BASE/'results/sft-pinning-driver-20261020-01'
DEST = BASE/'results/sft-pinning-verified-20261020'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    status = read(DRIVER/'status.json')
    if status['status'] != 'completed':
        raise ValueError('Do not review active, partial, or failed campaigns as completed')
    checked = verify(DRIVER/'sft-complete-key-results.zip',DRIVER/'independent-final-key-verification.json',64*1024**2)
    DEST.mkdir(exist_ok=False)
    with zipfile.ZipFile(DRIVER/'sft-complete-key-results.zip') as archive:
        archive.extractall(DEST)  # verify() checked every member/path/size/SHA above.
    root = DEST/'sft-pinning-validation-20261020-01'
    controller = read(root/'controller.json')
    if controller['status'] != 'completed' or any(x['exit_code'] for x in controller['commands']):
        raise ValueError('The actual controller did not complete all stages')
    expected = read(BASE/'results/sft-pinning-source-manifest-20261020.json')['files']
    if expected != controller['source_sha256']:
        raise ValueError('Frozen SFT sources differ')
    for name,sha in expected.items():
        path = BASE/name if name.endswith('.sh') else root/'source'/name
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen source bytes differ: '+name)
    primary = read(root/'numeric/primary-archive-verification.json')
    semantic = read(root/'semantics/semantics-verification.json')
    if (not primary['passed'] or not primary['checkpoint_resume_tested'] or not primary['primary_complete_raw_values_verified']
        or not semantic['passed'] or semantic['objective']!='answer_only_SFT' or not semantic['sft_mask_gate']):
        raise ValueError('Full SFT numeric/raw and semantic/native-resume gates failed')
    for arm in ('native','control','optimized'):
        source = read(DEST/'runs'/f'sft-pinning-audit-{arm}-20261020-01'/'sft-admission-verification.json')
        if not source['passed'] or not source['answer_only_supervision_source_reconstructed']:
            raise ValueError('Independent original-document SFT reconstruction failed')
    rounds = []
    for name in ('primary','confirmation') if controller['repeat_performed'] else ('primary',):
        result = analyze(DEST,'sft20261020'+name)
        actual = read(root/('formal-'+name+'.json'))
        for key in ('vs_native','vs_control','seed_mean_cv','repeat_required','round_promotion_passed'):
            if result[key] != actual[key]:
                raise ValueError('Locally recomputed timing/gate differs: '+key)
        if name == 'primary' and result['repeat_required'] and not controller['repeat_performed']:
            raise ValueError('Existing CV rule requires a complete confirmation round')
        arms = {}
        loss_checks = []
        for arm in ('native','control','optimized'):
            cases = [x for x in result['cases'] if x['arm']==arm]
            values = []
            for row in cases:
                folder = DEST/'runs'/row['case']
                case = read(folder/'case.json')
                objective = read(folder/'sft-objective-rank-0.json')
                rows = objective['supervision_rows']
                if case['objective']!='answer_only_SFT' or len(rows)!=120 or not objective['run_returned_successfully']:
                    raise ValueError('Actual formal objective or supervised batch coverage differs')
                other = next(x for x in result['cases'] if x['arm']=='native' and x['seed']==row['seed'])
                reference = read(DEST/'runs'/other['case']/'sft-objective-rank-0.json')['supervision_rows']
                if rows != reference:
                    raise ValueError('Formal SFT sample identity/supervision differs across paired arms')
                logs = [(folder/'stdout.log').read_text(encoding='utf-8'),
                        (DEST/'runs'/other['case']/'stdout.log').read_text(encoding='utf-8')]
                losses = [[float(value) for value in re.findall(r'lm loss:\s*([\d.E+\-]+)',text)] for text in logs]
                if any(len(values)!=12 or not all(math.isfinite(x) for x in values) for values in losses):
                    raise ValueError('All twelve formal logged loss intervals must be finite')
                errors = [abs(x-y) for x,y in zip(*losses)]
                consistent = all(abs(x-y)<=3e-4+1e-3*abs(y) for x,y in zip(*losses))
                if not consistent:
                    raise ValueError('Formal loss differs from paired native; investigate before accepting a speedup')
                loss_checks.append({'arm':arm,'seed':row['seed'],'intervals':12,
                                    'logged_loss_max_abs':max(errors),'passed':consistent,
                                    'scope':'posthoc aggregate loss check; no extra synchronization or GPU capture'})
                values.append(sum(x['supervised_answer_and_EOD_targets'] for x in rows[20:]))
            seconds = sum(100*x['mean_seconds_per_step'] for x in cases)
            arms[arm] = {'mean_seconds_per_step':statistics.mean(x['mean_seconds_per_step'] for x in cases),
                         'aggregate_input_token_slots_per_second':3*100*2048/seconds,
                         'aggregate_supervised_answer_targets_per_second':sum(values)/seconds}
        rounds.append({'round':name,'arms':arms,'formal_logged_loss_checks':loss_checks,
                       **{k:v for k,v in result.items() if k not in ('cases','round')}})
    passed = all(x['round_promotion_passed'] for x in rounds)
    if passed != controller['formal_performance_passed']:
        raise ValueError('Independent promotion differs from ACT')
    report = {'schema':'jiuge7b-sft-pinning-independent-completion-review-v1',
              'utc':datetime.now(timezone.utc).isoformat(),'all_validation_stages_completed':True,
              'formal_performance_passed':passed,'rounds':rounds,'repeat_performed':controller['repeat_performed'],
              'primary_full_numeric_and_resume_verified':True,'semantic_source_reconstruction_verified':True,
              'maximum_numeric_error':max(x['max_abs_error'] for x in primary['runs']),
              'ACT_unique_original_chunks_verified':primary['unique_raw_chunks_verified'],
              'verified_local_key_files':checked['verified_key_files'],'verified_local_key_bytes':checked['verified_key_bytes'],
              'local_large_raw_copy':False,'formal_throughput_unit':'input token slots/s;2048perstep;notTPsum',
              'old_CPT_84_4_percent_reused_as_SFT_result':False}
    with (BASE/'results/sft-pinning-independent-review-20261020.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    main()
