"""Version orchestration after the observed native wrapped-globals failure."""
from datetime import datetime, timezone
import base64
import hashlib
import json
from pathlib import Path
import subprocess

from drive_sft_pinning_validation import SSH, PREP

BASE = Path(__file__).resolve().parent
VERSIONS = (
    'sft_phased_moment_entry_20261023.py',
    'run_sft_first_step_phased_20261023.py',
    'run_sft_screen_phased_20261023.py',
    'probe_sft_runtime_phased_20261023.py',
    'admit_sft_first_step_phased_20261023.py',
    'continue_sft_screen_phased_20261023.py',
    'drive_sft_phased_moments_20261023.py',
    'review_sft_phased_moments_20261023.py',
    'package_sft_phased_moments_20261023.py',
    'plot_sft_phased_moments_20261023.py',
    'finish_sft_phased_moments_20261023.py',
)
RENAME = {name: name.replace('_20261023.py', '_v2_20261023.py') for name in VERSIONS}


def adapt(text):
    for before, after in RENAME.items():
        text = text.replace(before, after)
    # Python imports have no .py suffix.
    for before, after in RENAME.items():
        text = text.replace(before[:-3], after[:-3])
    return (text.replace('20261023-03', '20261023-07')
            .replace('sft_phased_moments_20261023.contract.json',
                     'sft_phased_moments_v2_20261023.contract.json'))


def write_new(name, text):
    target = BASE/name
    payload = text.encode()
    if target.exists():
        assert target.read_bytes() == payload, name
    else:
        with target.open('xb') as stream:
            stream.write(payload)


def main():
    prior_path = BASE/'results/sft-phased-moments-resume-source-manifest-20261023-04.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    for name, sha in prior['files'].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == sha, name
    cpu = json.loads((BASE/'results/phased-wrapped-initializer-CPU-preparation-20261023-06.json').read_text())
    assert cpu['passed']
    for name, sha in cpu['source_SHA256'].items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest() == sha
    for old, new in RENAME.items():
        text = adapt((BASE/old).read_text(encoding='utf-8'))
        if old == 'sft_phased_moment_entry_20261023.py':
            before = '    return text\n'
            assert text.count(before) == 1
            after = '''    for before,after in (
        ('from sft_local_moments_v3_20261021 import install','from sft_local_moments_v4_20261023 import install'),
        ('import sft_local_moments_v3_20261021 as moment_module','import sft_local_moments_v4_20261023 as moment_module'),
    ):
        if text.count(before)!=1:raise ValueError('Exact V4 candidate wiring required')
        text=text.replace(before,after)
    return text
'''
            text = text.replace(before, after)
        if old == 'probe_sft_runtime_phased_20261023.py':
            start = text.index('    env=os.environ.copy();')
            text = text[:start]+'''    results=[]
    for mode in ('0','1'):
        env=os.environ.copy();env.update(OMP_NUM_THREADS='8',JIUGE_FIXTURE_FUSED=mode,JIUGE_PHASE_FIXTURE_ROOT=str(root/('phased-wrapped-fused-'+mode)),JIUGE_PHASE_FIXTURE_HOST=socket.gethostname())
        with (root/('phased-wrapped-fused-'+mode+'.log')).open('x') as stream:
            subprocess.run([sys.executable,str(PREP/'check_sft_phased_wrapped_initializer_20261023.py')],check=True,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=260)
        result=json.loads((root/('phased-wrapped-fused-'+mode)/'summary.json').read_text())
        assert result['passed'] and not result['GPU_used']
        assert all(n['actual_thread_phase_switch_verified'] and n['native_disabled_initializer_exercised'] and n['fused']==(mode=='1') for n in result['nodes'])
        results.append(result)
    with (root/'phased-admission.json').open('x') as stream:json.dump({'passed':True,'GPU_used':False,'wrapped_fused_and_nonfused_fixtures':results},stream,indent=2)
    print(json.dumps({'passed':True,'actual_thread_phase_switch_verified':True,'wrapped_fused_and_nonfused_passed':True,'GPU_used':False}),flush=True)
'''
        write_new(new, text)
    bootstrap = (BASE/'bootstrap_sft_phased_moments_resume_20261023_04.sh').read_text()
    bootstrap = adapt(bootstrap).replace('logdir="$prep/sft-phased-moments-bootstrap-20261023-04"',
                                        'logdir="$prep/sft-phased-moments-bootstrap-20261023-07"')
    # The original new-window anchor is deliberately retained across retries.
    assert 'sft-phased-moments-bootstrap-20261023-04/campaign-first-start.epoch' in bootstrap
    write_new('bootstrap_sft_phased_moments_v2_20261023.sh', bootstrap)
    contract = json.loads((BASE/'sft_phased_moments_resume_20261023_04.contract.json').read_text())
    contract.update(stage='Wrapped native initializer CPU fused/nonfused passed; fresh GPU retry pending',
                    new_resource_authorization_received=True,
                    global_stop_before_epoch=1792771841,
                    initializer_repair='Use inspect.unwrap native globals and preserve observed Dynamo-disabled semantics; only original two moment factories change',
                    previous131_sources_unchanged=True,
                    previous_phased_first_update_failure_preserved=True)
    write_new('sft_phased_moments_v2_20261023.contract.json', json.dumps(contract, indent=2)+'\n')
    files = dict(prior['files'])
    additions = [*RENAME.values(), 'bootstrap_sft_phased_moments_v2_20261023.sh',
                 'sft_phased_moments_v2_20261023.contract.json', *cpu['source_SHA256']]
    assert len(additions) == 15
    for name in additions:
        assert name not in files
        files[name] = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
    for name in files:
        if name.endswith('.py'):
            compile((BASE/name).read_bytes(), name, 'exec')
    import sys
    sys.path.insert(0, str(BASE))
    from review_sft_phased_moments_v2_20261023 import generated
    compile(generated(), '<V4-fresh-review>', 'exec')
    manifest = {
        'schema': 'jiuge7b-phased-native-globals-fresh-source-freeze-v3',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'files': files, 'previous131_unchanged': True,
        'CPU_proof': cpu, 'global_stop_before_epoch': 1792771841,
        'maximum_total_seconds_across_new_window': 10800, 'max_charge_points': 0,
        'first_new_window_start_epoch': 1792761041,
        'fresh_GPU_first_update_completed': False,
        'formal_performance_evidence': False,
    }
    manifest_path = BASE/'results/sft-phased-moments-source-manifest-20261023-07.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())['files'] == files
    else:
        with manifest_path.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
    payload = {name: base64.b64encode((BASE/name).read_bytes()).decode() for name in additions}
    payload[manifest_path.name] = base64.b64encode(manifest_path.read_bytes()).decode()
    program = '''from pathlib import Path
import base64,hashlib,json,subprocess,sys
p=Path('''+repr(PREP)+''')
payload=json.loads('''+repr(json.dumps(payload))+''')
for name,value in payload.items():
 data=base64.b64decode(value);target=p/name
 if target.exists():assert target.read_bytes()==data,name
 else:
  with target.open('xb') as stream:stream.write(data)
manifest=json.loads((p/'''+repr(manifest_path.name)+''').read_text())
for name,expected in manifest['files'].items():
 f=p/name;assert hashlib.sha256(f.read_bytes()).hexdigest()==expected,name
 if name.endswith('.py'):compile(f.read_bytes(),name,'exec')
 elif name.endswith('.sh'):subprocess.run(['bash','-n',str(f)],check=True)
sys.path.insert(0,str(p))
from sft_phased_moment_entry_v2_20261023 import derive as entry
native=(p/'sft_numa_entry_20261021.py').read_bytes();expected=hashlib.sha256(native).hexdigest()
for steps in [1,35]:compile(entry(native,expected,steps),'<V4-native-entry>','exec')
from continue_sft_screen_phased_v2_20261023 import derive as controller
compile(controller((p/'continue_sft_numa_v2_20261021.py').read_bytes()),'<V4-six-controller>','exec')
from admit_sft_first_step_phased_v2_20261023 import derive as admit
compile(admit((p/'admit_sft_direct_zero_first_step_20261023.py').read_bytes()),'<V4-first-admission>','exec')
for name in ['sft-phased-first-step-20261023-07','runs/sft-phased-first-step-20261023-07','sft-phased-moments-screen-20261023-07','sft-phased-moments-approved-20261023-07.json']:
 assert not (p/name).exists(),name
assert 'torch' not in sys.modules
print(json.dumps({'passed':True,'source_count':len(manifest['files']),'manifest_sha256':hashlib.sha256((p/'''+repr(manifest_path.name)+''').read_bytes()).hexdigest(),'previous131_unchanged':True,'generated_programs_compile':True,'same_absolute_stop_epoch':1792771841,'GPU_used':False,'new_GPU_allocation_submitted':False}))
'''
    response = subprocess.run([*SSH, PREP+'/.venv/bin/python -'], input=program.encode(),
                              capture_output=True, timeout=180)
    if response.returncode:
        raise RuntimeError(response.stderr.decode('utf-8', errors='replace'))
    receipt = json.loads(response.stdout)
    assert receipt['source_count'] == 146
    assert receipt['manifest_sha256'] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with (BASE/'results/phased-native-globals-ACT-preparation-20261023-07.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
