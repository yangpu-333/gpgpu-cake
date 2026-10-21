"""CPU-only preservation of an OOM-interrupted screen; never a success receipt."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pinning_key_results as keys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
CAMPAIGN = PREP/'sft-moment-screen-20261021-01'
ROOT = PREP/'sft-moment-screen-failure-20261021-01'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def main():
    # No Torch import, GPU query, training, source mutation or resource creation.
    failure_path = PREP/'sft-moment-screen-platform-failure-20261021.json'
    failure = json.loads(failure_path.read_text())
    assert failure['platform_reason'] == 'OOMKilled' and failure['exit_code'] == 137
    assert failure['charge_points'] == 0 and not failure['six_case_screen_completed']
    assert failure['job'] == 'sg-huids25-261021-95919'
    state = json.loads((CAMPAIGN/'controller.json').read_text())
    assert [(x['policy'], x['tag']) for x in state['cases']] == [('baseline','a1'),('affinity-control','b1')]
    assert state['active_case'] == 'sft-moment-screen-20261021-c1-local-moments'
    receipt_path = ROOT/'verification.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        assert receipt['preservation_passed'] and digest(Path(receipt['bundle'])) == receipt['bundle_sha256']
        print(json.dumps(receipt))
        return
    ROOT.mkdir(exist_ok=False)
    approval = json.loads((CAMPAIGN/'approval.json').read_text())
    manifest = json.loads((PREP/'sft-moment-screen-source-manifest-v2-20261021.json').read_text())['files']
    assert len(manifest) == 73 and manifest == approval['source_files_sha256']
    for name, sha in manifest.items():
        assert digest(PREP/name) == digest(CAMPAIGN/'source'/name) == sha, name
    for name in ('sft-moment-screen-platform-failure-20261021.json',
                 'sft-local-moments-R2-OOM-zero-points-20261021.png',
                 'sft-moment-screen-local-observers-stopped-20261021.json'):
        shutil.copyfile(PREP/name, ROOT/name)
    shutil.copyfile(Path(__file__), ROOT/Path(__file__).name)
    cases, traces, folders = [], [], []
    paired = None
    for item in state['cases']:
        folder = PREP/'runs'/item['case']
        completion = json.loads((folder/'completion-verification.json').read_text())
        assert completion['passed'] and completion['actual_non_skipped_updates_each_rank'] == 35
        assert completion['adam_state_entries_each_rank'] == 199
        rows = json.loads((folder/'sft-supervision-rank-0.json').read_text())['rows']
        identity = (rows, completion['native_losses'])
        assert len(rows) == len(completion['native_losses']) == 35
        if paired is None:
            paired = identity
        else:
            assert paired == identity
        for script, name in [('review_native_trace.py','trace-analysis.json'),
                             ('inspect_trace_communication.py','communication-analysis.json'),
                             ('analyze_sft_paired_trace_20261020.py','symmetric-mechanism.json')]:
            target = folder/name
            if target.exists():
                raise ValueError('Unexpected closed analysis; inspect before versioned recovery: '+str(target))
            with (ROOT/(item['tag']+'-'+name+'.log')).open('x') as stream:
                subprocess.run([sys.executable,str(PREP/script),'--case-root',str(folder),'--output',str(target)],
                               stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=300)
        mechanism = json.loads((folder/'symmetric-mechanism.json').read_text())
        assert len(mechanism['windows']) == 4
        windows = []
        for window, steps in [('window-01',list(range(21,26))),('window-02',list(range(31,36)))]:
            durations = []
            for rank in (0,1):
                cap = json.loads((folder/window/f'rank-{rank}-capture.json').read_text())
                raw = folder/window/f'rank-{rank}.json.gz'
                assert cap['stable_step_ids'] == steps and digest(raw) == cap['trace_sha256']
                h = hashlib.sha256()
                with gzip.open(raw,'rb') as stream:
                    while block := stream.read(1024*1024): h.update(block)
                assert h.hexdigest() == cap['decompressed_json_sha256']
                observation = next(x for x in mechanism['windows'] if x['window'] == window and x['rank'] == rank)
                assert observation['host_ranges']['jiuge/cpu_adamw_math']['count'] == 995
                durations.append(observation['window_seconds'])
                traces.append({'path':raw.relative_to(PREP).as_posix(),'rank':rank,'window':window,
                               'compressed_sha256':digest(raw),'decompressed_sha256':h.hexdigest()})
            windows.append(max(durations)/5)
        subprocess.run([sys.executable,str(PREP/'archive_case.py'),'--case-root',str(folder),
                        '--create','--snapshot-analysis'],check=True,stdout=subprocess.DEVNULL)
        archive_receipt = ROOT/(item['case']+'-archive-verification.json')
        subprocess.run([sys.executable,str(PREP/'archive_case.py'),'--case-root',str(folder),
                        '--verify','--receipt',str(archive_receipt)],check=True,stdout=subprocess.DEVNULL)
        cases.append({**item,'completed_updates_each_rank':35,'verified_raw_traces':4,
                      'diagnostic_window_slow_rank_seconds_per_step':windows,
                      'diagnostic_mean_seconds_per_step':sum(windows)/2,
                      'full_candidate_comparison_available':False})
        folders.append(folder)
    failed = PREP/'runs'/state['active_case']
    assert not (failed/'case.json').exists() and not (failed/'sft-numa-after-first-update-rank-0.json').exists()
    assert not list(failed.glob('window-*'))
    assert len(traces) == 8
    summary = {'schema':'jiuge7b-sft-moment-interrupted-screen-preservation-v1',
               'verified_utc':datetime.now(timezone.utc).isoformat(),'platform_failure':failure,
               'frozen_sources_verified':73,'completed_cases':cases,'verified_raw_traces':8,
               'paired35_samples_counts_logged_losses_equal_between_completed_controls':True,
               'candidate_first_update_completed':False,'candidate_comparison_available':False,
               'full_six_case_screen_passed':False,'formal_performance_evidence':False,
               'new_7B_full_state_or_resume_acceptance':False,
               'OOM_allocator_or_NUMA_cause_isolated':False,'GPU_used_for_preservation':False}
    write(ROOT/'partial-review.json',summary)
    selected = {}
    for folder in [CAMPAIGN,*folders,failed,ROOT]:
        for path in folder.rglob('*'):
            if path.is_file(): selected[path.relative_to(PREP).as_posix()] = path
    files = {name:{'bytes':path.stat().st_size,'sha256':digest(path)} for name,path in sorted(selected.items())}
    raw_manifest = {'schema':'jiuge7b-sft-moment-failed-raw-bundle-v1','files':files,
                    'traces':traces,'screen_completed':False,'platform_failure':failure,
                    'all_raw_remains_ACT':True,'formal_performance_evidence':False}
    bundle = ROOT/'Jiuge7B-SFT-local-moments-interrupted-OOM-evidence.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_STORED) as archive:
        for name,path in selected.items(): archive.write(path,name)
        archive.writestr('failure-delivery-manifest.json',json.dumps(raw_manifest,indent=2)+'\n')
    with zipfile.ZipFile(bundle) as archive:
        for name,entry in files.items():
            with archive.open(name) as stream:
                assert hashlib.file_digest(stream,'sha256').hexdigest() == entry['sha256'], name
        for trace in traces:
            with archive.open(trace['path']) as source,gzip.GzipFile(fileobj=source) as decoded:
                assert hashlib.file_digest(decoded,'sha256').hexdigest() == trace['decompressed_sha256']
    write(ROOT/'raw-manifest.json',raw_manifest)
    verification = {'schema':'jiuge7b-sft-moment-failure-preservation-verification-v1',
                    'preservation_passed':True,'scientific_screen_completed':False,
                    'bundle':str(bundle),'bundle_bytes':bundle.stat().st_size,'bundle_sha256':digest(bundle),
                    'verified_files':len(files),'verified_raw_traces':8,'completed_cases':2,
                    'compressed_and_decompressed_SHA_passed':True,'raw_downloaded_to_local':False}
    write(receipt_path,verification)
    keys.KEY_SUFFIXES = keys.KEY_SUFFIXES | {'.sh'}
    key_bundle = PREP/'sft-moment-screen-failure-keys-20261021.zip'
    catalog = keys.export(PREP,[CAMPAIGN,*folders,failed,ROOT],key_bundle,32*1024**2)
    result = {**verification,'key_bundle':str(key_bundle),'key_bundle_bytes':key_bundle.stat().st_size,
              'key_bundle_sha256':digest(key_bundle),'key_files':catalog['key_file_count'],
              'key_uncompressed_bytes':catalog['key_uncompressed_bytes']}
    write(ROOT/'key-bundle.json',result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
