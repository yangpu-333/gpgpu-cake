"""Stage only new versioned sources and freeze old accepted files unchanged."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from drive_sft_pinning_validation import SSH,SCP,PREP,run

BASE = Path(__file__).resolve().parent
NEW = ('sft_trace_ranges_20261020.py','sft_trace_entry_20261020.py',
       'run_sft_trace_case_20261020.py','analyze_sft_paired_trace_20261020.py',
       'probe_sft_trace_ranges_20261020.py','continue_sft_paired_trace_20261020.py',
       'bootstrap_sft_paired_trace_20261020.sh','sft_paired_trace_20261020.contract.json')
COMMON = ('review_native_trace.py','analyze_trace.py','inspect_copy_api.py',
          'inspect_trace_communication.py','archive_case.py','pinning_key_results.py',
          'verify_trace_case_completion.py')


def main():
    old = json.loads((BASE/'results/sft-pinning-formal-source-manifest-v3-20261020.json').read_text(encoding='utf-8'))['files']
    files = dict(old)
    for name in [*old,*NEW,*COMMON]:
        data = (BASE/name).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if name in old and old[name] != sha:
            raise ValueError('Old frozen source changed: '+name)
        if name.endswith('.py'):
            compile(data,name,'exec')
        files[name] = sha
    target = BASE/'results/sft-paired-trace-source-manifest-20261020.json'
    manifest = {'schema':'jiuge7b-sft-paired-trace-source-manifest-v1','files':files,
                'prior31_scientific_sources_unchanged':True,'new_versioned_files':list(NEW)}
    with target.open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2)
    run([*SCP,*[str(BASE/name) for name in NEW],'huids25@192.168.5.200:'+PREP+'/'])
    code = '''import hashlib,json,subprocess,sys
from pathlib import Path
p=Path('''+repr(PREP)+''')
files='''+repr(files)+'''
for n,s in files.items():
    assert hashlib.sha256((p/n).read_bytes()).hexdigest()==s,n
for ext in ('.bin','.idx'):
    assert (p/('datasets/alpaca-sft-answer-v1'+ext)).is_file()
subprocess.run(['bash','-n',str(p/'bootstrap_sft_paired_trace_20261020.sh')],check=True)
probe=subprocess.check_output([sys.executable,str(p/'probe_sft_trace_ranges_20261020.py')],text=True)
print(json.dumps({'passed':True,'source_files_verified':len(files),'old31_unchanged':True,'CPU_probe':json.loads(probe)}))
'''
    proof = json.loads(run([*SSH,PREP+'/.venv/bin/python -'],input=code.encode()).decode())
    proof['utc'] = datetime.now(timezone.utc).isoformat()
    with (BASE/'results/sft-paired-trace-preparation-20261020.json').open('x',encoding='utf-8') as stream:
        json.dump(proof,stream,indent=2)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
