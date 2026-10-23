"""Freeze and deploy a pending budget amendment; no resource or GPU launch."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from drive_sft_pinning_validation import SSH, PREP

BASE = Path(__file__).resolve().parent
MANIFEST = BASE/'results/sft-phased-moments-resume-source-manifest-20261023-04.json'
RECEIPT = BASE/'results/sft-phased-moments-resume-ACT-preparation-20261023-04.json'
ADDITIONS = (
    'bootstrap_sft_phased_moments_resume_20261023_04.sh',
    'sft_phased_moments_resume_20261023_04.contract.json',
    'review_sft_phased_moments_resume_20261023_04.py',
    'finish_sft_phased_moments_resume_20261023_04.py',
)


def main():
    previous_path = BASE/'results/sft-phased-moments-source-manifest-20261023-03.json'
    previous = json.loads(previous_path.read_text())
    files = dict(previous['files'])
    assert len(files) == 127
    for name, expected in files.items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == expected, name
    for name in ADDITIONS:
        assert name not in files
        files[name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    for name in files:
        if name.endswith('.py'):
            compile((BASE/name).read_bytes(), name, 'exec')
    from review_sft_phased_moments_resume_20261023_04 import generated
    compile(generated(), '<pending-budget-independent-review>', 'exec')
    manifest = {
        'schema': 'jiuge7b-phased-moments-pending-budget-source-freeze-v2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': files, 'previous_127_unchanged': True,
        'previous_manifest_sha256': hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        'CPU_five_step_and_native_resume_passed': True,
        'new_7B_first_step_completed': False,
        'new_resource_authorization_pending': True,
        'resource_submitted': False, 'max_charge_points': 0,
        'maximum_new_window_seconds_including_setup_and_retries': 10800,
        'old_window_expired_at': '2026-10-23T12:55:28+08:00',
    }
    if MANIFEST.exists():
        actual = json.loads(MANIFEST.read_text())
        assert actual['files'] == files
        manifest = actual
    else:
        with MANIFEST.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
    payload = {name: base64.b64encode((BASE/name).read_bytes()).decode() for name in files}
    payload[previous_path.name] = base64.b64encode(previous_path.read_bytes()).decode()
    payload[MANIFEST.name] = base64.b64encode(MANIFEST.read_bytes()).decode()
    preserve = BASE/'preserve_sft_direct_zero_20261023.py'
    payload[preserve.name] = base64.b64encode(preserve.read_bytes()).decode()
    program = '''from pathlib import Path
from datetime import datetime,timezone
import base64,hashlib,json,subprocess,sys
p=Path('''+repr(PREP)+''')
payload=json.loads('''+repr(json.dumps(payload))+''')
for name,encoded in payload.items():
    assert Path(name).name==name
    data=base64.b64decode(encoded)
    target=p/name
    if target.exists():assert target.read_bytes()==data,name
    else:
        with target.open('xb') as stream:stream.write(data)
manifest=json.loads((p/'''+repr(MANIFEST.name)+''').read_text())
for name,expected in manifest['files'].items():
    path=p/name
    assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    if name.endswith('.py'):compile(path.read_bytes(),name,'exec')
    elif name.endswith('.sh'):subprocess.run(['bash','-n',str(path)],check=True)
sys.path.insert(0,str(p))
from review_sft_phased_moments_resume_20261023_04 import generated
compile(generated(),'<pending-window-review>','exec')
from sft_phased_moment_entry_20261023 import derive as entry
native=(p/'sft_numa_entry_20261021.py').read_bytes()
expected=json.loads((p/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']['sft_numa_entry_20261021.py']
for updates in (1,35):compile(entry(native,expected,updates),'<phased-entry>','exec')
from admit_sft_first_step_phased_20261023 import derive as admit
compile(admit((p/'admit_sft_direct_zero_first_step_20261023.py').read_bytes()),'<phased-first-admission>','exec')
from continue_sft_screen_phased_20261023 import derive as controller
compile(controller((p/'continue_sft_numa_v2_20261021.py').read_bytes()),'<phased-six-controller>','exec')
uncreated=['sft-phased-first-step-20261023-03','runs/sft-phased-first-step-20261023-03','sft-phased-moments-screen-20261023-03','sft-phased-moments-approved-20261023-03.json','sft-phased-moments-bootstrap-20261023-04']
for name in uncreated:assert not (p/name).exists(),name
result={'schema':'jiuge7b-phased-moments-pending-window-ACT-preparation-v2','passed':True,'verified_utc':datetime.now(timezone.utc).isoformat(),'source_files':len(manifest['files']),'manifest_sha256':hashlib.sha256((p/'''+repr(MANIFEST.name)+''').read_bytes()).hexdigest(),'all_local_ACT_source_bytes_equal':True,'Python_and_Bash_syntax_passed':True,'generated_first_step_six_case_and_review_programs_compile':True,'previous_127_unchanged':True,'future_roots_and_approval_still_uncreated':True,'CUDA_initialized':False,'GPU_resource_submitted':False,'new_resource_authorization_pending':True,'morning_window_expired':True}
assert 'torch' not in sys.modules
print(json.dumps(result),flush=True)
'''
    response = subprocess.run([*SSH, PREP+'/.venv/bin/python -'], input=program.encode(),
                              capture_output=True, check=True, timeout=180)
    result = json.loads(response.stdout)
    assert result['source_files'] == 131 and result['passed']
    assert result['manifest_sha256'] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    with RECEIPT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
