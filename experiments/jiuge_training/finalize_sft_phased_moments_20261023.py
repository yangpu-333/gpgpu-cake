"""Archive the reviewed phased-moment diagnostic and two human reports on ACT."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from drive_sft_pinning_validation import SSH, SCP, PREP, run

BASE = Path(__file__).resolve().parent
STAGE = BASE/'results/sft-phased-final-delivery-20261023-07'
REMOTE = PREP+'/sft-phased-moments-delivery-20261023-07/final-review'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')


def main():
    review = read(BASE/'results/sft-phased-moments-independent-review-20261023-07.json')
    raw = read(BASE/'results/sft-phased-moments-delivery-verification-20261023-07.json')
    mechanism = read(BASE/'results/sft-phased-moments-mechanism-summary-20261023-07.json')
    post = read(BASE/'results/sft-phased-moments-postprocess-20261023-07/status.json')
    platform = read(BASE/'results/sft-phased-moments-platform-completion-20261023-07.json')
    if (not review['passed'] or review['formal_performance_passed']
            or len(review['cases']) != 6 or not raw['passed'] or raw['verified_raw_traces'] != 24
            or post['status'] != 'scientific_postprocessing_completed_reports_pending'
            or platform['job'] != 'sg-huids25-261023-24c7a'
            or platform['status'] != 'completed' or platform['points'] != 0
            or platform['controller_exit_code'] != 0
            or not mechanism['passed'] or mechanism['formal_performance_evidence']
            or len(mechanism['windows']) != 24):
        raise ValueError('Close scientific review, raw SHA and zero-cost resource gates first')
    if sha(BASE/'summarize_sft_phased_mechanism_20261023.py') != mechanism['script_sha256']:
        raise ValueError('Supplementary CPU summary source changed')
    manifest = read(BASE/'results/sft-phased-moments-source-manifest-20261023-07.json')['files']
    for name, expected in manifest.items():
        if sha(BASE/name) != expected:
            raise ValueError('Frozen local source changed: '+name)
    for name, expected in post['CPU_source_sha256'].items():
        if sha(BASE/name) != expected:
            raise ValueError('CPU postprocessing source changed: '+name)
    code = '''from pathlib import Path
import json,hashlib
p=Path('''+repr(PREP)+''');files='''+repr(manifest)+'''
for name,expected in files.items():
    for f in (p/name,p/'sft-phased-moments-screen-20261023-07/source'/name):
        assert hashlib.sha256(f.read_bytes()).hexdigest()==expected,str(f)
print(json.dumps({'passed':True,'files':len(files),'ACT_live_and_execution_snapshot_SHA_match':True,'original131_unchanged':True}))
'''
    proof = json.loads(run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode()))
    write(BASE/'results/sft-phased-moments-final-source-verification-20261023-07.json', proof)
    selected = [BASE/'README.md', BASE/'BRIEF.md', Path(__file__),
                BASE/'summarize_sft_phased_mechanism_20261023.py',
                BASE/'results/sft-phased-moments-platform-completed-20261023-07.jpg']
    selected += [BASE/name for name in manifest]
    selected += [BASE/name for name in post['CPU_source_sha256']]
    for pattern in ('*20261023*.json', 'sft-phased-moments-*20261023-07.png', 'sft-phased-moments-*20261023-07.svg'):
        selected += list((BASE/'results').glob(pattern))
    selected += list((BASE/'results').glob('sft-phased-finalization-network-failure-20261023-*.json'))
    for folder in ('sft-phased-moments-driver-20261023-03','sft-phased-moments-postprocess-20261023-03',
                   'sft-phased-moments-driver-20261023-07','sft-phased-moments-postprocess-20261023-07'):
        root=BASE/'results'/folder
        selected += [f for f in root.iterdir() if f.is_file() and f.suffix in ('.json','.log','.zip')]
    # Include small prior artifacts linked by the two reports, not their traces.
    for name in ('sft-paired-trace-comparison-20261020.png', 'sft-paired-trace-lanes-20261020.png',
                 'sft-paired-trace-independent-review-v2-20261020.json',
                 'sft-paired-trace-delivery-verification-20261020.json',
                 'sft-pinning-independent-review-20261020-continuation-v2.json'):
        selected.append(BASE/'results'/name)
    selected = list({str(p.resolve()): p for p in selected}.values())
    if any(not p.is_file() for p in selected):
        raise ValueError('Final key artifact missing')
    if sum(p.stat().st_size for p in selected) > 32*1024**2:
        raise ValueError('Bounded final-review budget exceeded')
    STAGE.mkdir(exist_ok=False)
    files = {}
    for source in selected:
        relative = source.resolve().relative_to(BASE.resolve())
        destination = STAGE/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        files[relative.as_posix()] = {'bytes': source.stat().st_size, 'sha256': sha(source)}
    catalog = {'schema': 'jiuge7b-sft-phased-moments-final-review-catalog-v1', 'files': files,
               'human_reports': ['README.md', 'BRIEF.md'], 'raw_zip_downloaded': False,
               'raw_ACT_bundle_SHA256': raw['bundle_sha256'], 'platform': platform,
               'diagnostic_only': True, 'new_formal_performance_passed': False,
               'all_prior_failures_preserved': True}
    write(STAGE/'manifest.json', catalog)
    code = 'from pathlib import Path;p=Path('+repr(REMOTE)+');assert not p.exists();print("new-final-review-target-absent")'
    run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode())
    run([*SCP, '-r', str(STAGE), 'huids25@192.168.5.200:'+REMOTE])
    code = '''from pathlib import Path
import json,hashlib
p=Path('''+repr(REMOTE)+''');m=json.loads((p/'manifest.json').read_text())
for name,expected in m['files'].items():
    f=p/name
    assert f.stat().st_size==expected['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==expected['sha256'],name
r={'passed':True,'folder':str(p),'files':len(m['files']),'bytes':sum(x['bytes'] for x in m['files'].values()),
   'manifest_sha256':hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest(),'originals_modified':False}
with p.with_name(p.name+'.verification.json').open('x') as stream:json.dump(r,stream,indent=2)
print(json.dumps(r))
'''
    verified = json.loads(run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode()))
    if verified['manifest_sha256'] != sha(STAGE/'manifest.json'):
        raise ValueError('Final ACT/local catalog SHA differs')
    write(BASE/'results/sft-phased-moments-final-ACT-review-verification-20261023-07.json', verified)
    completion = {'schema': 'jiuge7b-sft-phased-moments-completion-v1', 'utc': datetime.now(timezone.utc).isoformat(),
                  'requested_diagnostic_completed': True, 'fresh_cases': 6, 'raw_trace_count': 24,
                  'diagnostic_baseline': 'accepted pinned SFT repair; fresh unbound runs',
                  'candidate_screens': review['candidate_screens'],
                  'new_full_state_resume_and_formal_still_required_for_any_candidate': True,
                  'new_formal_performance_passed': False, 'ACT_raw_archive': raw,
                  'ACT_final_review': verified, 'platform': platform,
                  'raw_originals_remain_ACT': True, 'local_only_bounded_keys': True,
                  'two_human_reports': ['README.md', 'BRIEF.md'], 'keep_old_V100': True}
    output = BASE/'results/sft-phased-moments-completion-receipt-20261023-07.json'
    write(output, completion)
    remote_receipt = PREP+'/sft-phased-moments-delivery-20261023-07/final-review-completion.json'
    run([*SCP, str(output), 'huids25@192.168.5.200:'+remote_receipt])
    code = 'from pathlib import Path;import hashlib;print(hashlib.sha256(Path('+repr(remote_receipt)+').read_bytes()).hexdigest())'
    receipt_sha = run([*SSH, PREP+'/.venv/bin/python -'], input=code.encode()).decode().strip()
    if receipt_sha != sha(output):
        raise ValueError('Completion receipt ACT/local SHA differs')
    write(BASE/'results/sft-phased-moments-completion-upload-verification-20261023-07.json',
          {'passed': True, 'ACT_path': remote_receipt, 'sha256': receipt_sha})
    active = BASE/'results/active-validation-state.json'
    state = read(active)
    state.update(as_of_local=datetime.now().astimezone().isoformat(), experiments_running_for_this_campaign=False,
                 resource='24c7a completed0points; oldV100 retained',
                 current_actual_progress='All six NUMA diagnostic cases,24traces, independent review, ACT raw/archive and two reports closed',
                 next_actions=['No active GPU work; any NUMA candidate requires its own full SFT acceptance'])
    state['sft_phased_moments_20261023'].update(status='completed', job_status='completed', platform_points=0,
        verified_raw_traces=24, completion_receipt=str(output.relative_to(BASE)).replace('\\', '/'),
        reports_complete=True, final_ACT_review=verified, candidate_screens=review['candidate_screens'],
        raw_package=raw['bundle'], raw_zip_local_downloaded=False, new_formal_performance_passed=False,
        heartbeat='completion closed; pause automation next')
    active.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'ACT_final_review': verified, 'receipt_SHA': receipt_sha}, indent=2))


if __name__ == '__main__':
    main()
