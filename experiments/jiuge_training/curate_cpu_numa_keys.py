"""Export compact summaries; full page samples/numa_maps and all raw stay ACT."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from pinning_key_results import export

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('diagnostic','validation'),required=True)
    args=parser.parse_args()
    stage=args.stage
    root=PREP/f'cpu-numa-curated-{stage}-keys-20261016-01'; root.mkdir(exist_ok=False)
    copied={}
    def copy(path):
        if not path.is_file() or path.is_symlink():
            raise ValueError('Named regular key file required: '+str(path))
        relative=path.relative_to(PREP); target=root/'keys'/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,target)
        copied[str(relative)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if stage=='diagnostic':
        main_root=PREP/'cpu-numa-mechanism-20261016-01'
        if json.loads((main_root/'controller.json').read_text())['status']!='completed_diagnostic_analysis':
            raise ValueError('Finish all closed-trace analysis first')
        for folder in (main_root,PREP/'cpu-numa-adamw-review-20261016-01'):
            for path in folder.rglob('*'):
                if path.is_file() and path.suffix in {'.json','.py','.log'}:
                    copy(path)
        control=json.loads((PREP/'cpu-numa-diagnostics-20261016-02/controller.json').read_text())
        for folder_name in ('cpu-numa-diagnostics-20261016-01','cpu-numa-diagnostics-20261016-02'):
            folder=PREP/folder_name
            for name in ('controller.json',):
                copy(folder/name)
            for path in folder.glob('*primary-verification.json'):
                copy(path)
        cases=[row['case'] for row in control['cases']]+['cpu-numa-local8-02']
    else:
        main_root=PREP/'cpu-numa-validation-20261016-01'
        if json.loads((main_root/'controller.json').read_text())['status']!='ready_for_local_key_verification':
            raise ValueError('Finish complete new candidate validation first')
        for path in main_root.rglob('*'):
            if (path.is_file() and path.suffix in {'.json','.py','.patch','.log'} and
                not path.name.startswith('numa-') and path.name!='tensor-journal.jsonl'):
                copy(path)
        cases=['cpu-numa-local8-full-audit-01','cpu-numa-local8-full-resume-01']
    for case in cases:
        folder=PREP/'runs'/case
        for name in ('case-start.json','case.json','stdout.log','stderr.log','archive-manifest.json',
                     'numa-binding-rank-0.json','numa-binding-rank-1.json'):
            if (folder/name).is_file():
                copy(folder/name)
        for pattern in ('rank-*/entry.json','rank-*/optimizer-state.json','rank-*/weight-loading.json','window-*/rank-*-capture.json'):
            for path in folder.glob(pattern):
                copy(path)
    copy(PREP/'cpu_numa.contract.json'); copy(Path(__file__))
    (root/'source-file-inventory.json').write_text(json.dumps({'source_files':copied,'full_raw_originals_download_enabled':False,
          'full_numa_page_samples_and_proc_maps_stay_ACT':True},indent=2)+'\n')
    bundle=PREP/f'cpu-numa-curated-{stage}-key-results-20261016.zip'
    catalog=export(PREP,[root],bundle)
    print(json.dumps({'bundle':str(bundle),'key_files':catalog['key_file_count'],'key_bytes':catalog['key_uncompressed_bytes']}),flush=True)


if __name__=='__main__':
    main()
