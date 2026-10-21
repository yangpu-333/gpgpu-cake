"""Close an already completed CPT/SFT capture and export only bounded keys.

Run on ACT CPU after the GPU controller has finished. Full traces stay on ACT.
"""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from pinning_key_results import export

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT=PREP/'sft-tracing-20261017-01'
DELIVERY=PREP/'tracing-delivery-20261017-02'
CASES=('sft-native-admission-20261017-01','sft-native-profile-20261017-01',
       'cpt-native-admission-20261017-01','cpt-native-profile-20261017-01')


def read(path):
    return json.loads(path.read_text())


def stream_sha(stream):
    value=hashlib.sha256()
    while block:=stream.read(1024*1024):
        value.update(block)
    return value.hexdigest()


def main():
    controller=read(ROOT/'controller.json')
    if controller['status']!='completed' or controller['completed_cases']!=list(CASES):
        raise ValueError('All actual cases must be complete before closing; do not export active results')
    if any(command['exit_code'] for command in controller['commands']):
        raise ValueError('A controller stage failed')
    summary=read(ROOT/'tracing-summary.json')
    if not summary['passed'] or summary['formal_speedup_evidence']:
        raise ValueError('Verified diagnostic summary without a speedup claim required')
    expected=read(PREP/'sft-source-manifest-v4-20261017.json')['files']
    actual=read(ROOT/'source-sha256.json')
    if actual!=expected:
        raise ValueError('Actual frozen capture sources differ from the cross-checked manifest')
    cases=[]
    selected={}
    traces=[]
    for name in CASES:
        folder=PREP/'runs'/name
        case=read(folder/'case.json')
        completion=read(folder/'completion-verification.json')
        archive=read(ROOT/(name+'-archive-verification.json'))
        with (folder/'archive-manifest.json').open('rb') as stream:
            manifest_sha=stream_sha(stream)
        if (case['exit_code'] or not completion['passed'] or not archive['passed']
            or archive['manifest_sha256']!=manifest_sha):
            raise ValueError('Actual completion or ACT archive failed: '+name)
        if case['resource_gate_before'].strip() or case['resource_gate_after'].strip():
            raise ValueError('Unexpected compute activity at an assigned-GPU boundary: '+name)
        cases.append({'case':name,'steps_each_rank':completion['actual_non_skipped_updates_each_rank'],
                      'optimizer_steps_each_rank':completion['all_adam_states_step'],
                      'ACT_original_files_verified':archive['verified_file_count'],
                      'ACT_original_bytes':archive['total_expected_bytes']})
        if 'profile' not in name:
            continue
        for filename in ('case.json','case-start.json','stdout.log','stderr.log','completion-verification.json',
                         'trace-analysis.json','communication-analysis.json','archive-manifest.json'):
            path=folder/filename
            selected[path.relative_to(PREP).as_posix()]=path
        for path in (folder/'source').glob('*.py'):
            selected[path.relative_to(PREP).as_posix()]=path
        for rank in (0,1):
            for filename in ('entry.json','weight-loading.json','optimizer-state.json'):
                path=folder/f'rank-{rank}'/filename
                selected[path.relative_to(PREP).as_posix()]=path
        for window in (1,2):
            for rank in (0,1):
                path=folder/f'window-{window:02d}/rank-{rank}.json.gz'
                receipt_path=path.with_name(f'rank-{rank}-capture.json')
                receipt=read(receipt_path)
                with path.open('rb') as stream:
                    compressed=stream_sha(stream)
                with gzip.open(path,'rb') as stream:
                    decompressed=stream_sha(stream)
                if compressed!=receipt['trace_sha256'] or decompressed!=receipt['decompressed_json_sha256']:
                    raise ValueError('Actual trace hash differs')
                for original in (path,receipt_path):
                    selected[original.relative_to(PREP).as_posix()]=original
                traces.append({'case':name,'window':window,'rank':rank,'steps':receipt['stable_step_ids'],
                    'path':path.relative_to(PREP).as_posix(),'compressed_sha256':compressed,
                    'decompressed_sha256':decompressed,'kernels':receipt['gpu_kernel_events'],
                    'NCCL_kernels':receipt['gpu_collective_kernel_events']})
    if len(traces)!=8 or not summary['actual_same_input_admission']['passed']:
        raise ValueError('Eight real native CPT/SFT traces and actual input-pair verification required')
    DELIVERY.mkdir(exist_ok=False)
    shutil.copyfile(__file__,DELIVERY/'close_cpt_sft_delivery.py')
    shutil.copyfile(ROOT/'tracing-summary.json',DELIVERY/'tracing-summary.json')
    files={}
    for name,path in sorted(selected.items()):
        with path.open('rb') as stream:
            files[name]={'bytes':path.stat().st_size,'sha256':stream_sha(stream)}
    catalog={'schema':'jiuge7b-native-CPT-SFT-trace-delivery-v1','files':files,'traces':traces,
             'same_allocation':'sg-huids25-261017-e2bcb / xfusion-2 / PIX / 2A100',
             'CPT_and_SFT_actual_35step_training':True,'formal_speedup_evidence':False,
             'originals_in_archive':True,'large_tensor_states_or_model_weights_in_archive':False,
             'collector_drop_count_available':False,'full_causal_critical_path_proven':False}
    bundle=DELIVERY/'Jiuge7B-native-CPT-SFT-S2048-traces.zip'
    with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_STORED) as archive:
        for name,path in selected.items():
            archive.write(path,name)
        archive.writestr('trace-delivery-manifest.json',json.dumps(catalog,indent=2)+'\n')
    with zipfile.ZipFile(bundle) as archive:
        for name,record in files.items():
            with archive.open(name) as stream:
                if stream_sha(stream)!=record['sha256']:
                    raise ValueError('Archived raw delivery file differs')
        for trace in traces:
            with archive.open(trace['path']) as stream,gzip.GzipFile(fileobj=stream) as decompressed:
                if stream_sha(decompressed)!=trace['decompressed_sha256']:
                    raise ValueError('Packaged decompressed trace differs')
    with bundle.open('rb') as stream:
        digest=stream_sha(stream)
    result={'schema':'jiuge7b-CPT-SFT-closed-delivery-review-v1','utc':datetime.now(timezone.utc).isoformat(),
            'passed':True,'cases':cases,'source_files_verified':len(actual),'trace_windows':8,
            'actual_input_tokens_labels_positions_attention_equal_except_task_loss_mask':True,
            'full_model_training_each_task':True,'original_bundle':str(bundle),
            'original_bundle_bytes':bundle.stat().st_size,'original_bundle_sha256':digest,
            'original_bundle_files_verified':len(files),'traces':traces,
            'formal_speedup_evidence':False,'full_gradient_optimizer_equivalence_claimed':False,
            'originals_downloaded_to_local':False,'old_V100_modified_or_stopped':False}
    with (DELIVERY/'closed-review.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    key_result=export(PREP,[ROOT,*(PREP/'runs'/name for name in CASES),DELIVERY,
                           PREP/'sft-ubuntu-bootstrap-20261017-01'],
                      DELIVERY/'jiuge7b-CPT-SFT-key-results-20261017.zip',64*1024**2)
    with (DELIVERY/'key-export-summary.json').open('x') as stream:
        json.dump({key:value for key,value in key_result.items() if key not in ('files','excluded_originals')},stream,indent=2)
    print(json.dumps({'passed':True,'raw_trace_bundle_bytes':result['original_bundle_bytes'],
                      'raw_trace_bundle_sha256':digest,'key_files':key_result['key_file_count'],
                      'key_uncompressed_bytes':key_result['key_uncompressed_bytes']}),flush=True)


if __name__=='__main__':
    main()
