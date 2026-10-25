"""Freeze new formal orchestration; check existing closed evidence without GPUs."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from drive_sft_pinning_validation import PREP, SSH, SCP

BASE = Path(__file__).resolve().parent
FILES = ('sft_phased_formal_20261025.py','run_sft_phased_formal_20261025.py',
         'sft_phased_formal_entry_20261025.py','probe_sft_phased_formal_20261025.py',
         'continue_sft_phased_formal_20261025.py','bootstrap_sft_phased_formal_20261025.sh',
         'review_sft_phased_formal_20261025.py','drive_sft_phased_formal_20261025.py',
         'approve_sft_phased_formal_20261025.py','prepare_sft_phased_formal_20261025.py')


def main():
    old = json.loads((BASE/'results/sft-phased-acceptance-source-manifest-20261024-02.json').read_text())['files']
    assert len(old)==169
    for name, digest in old.items():
        assert hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest, name
    hashes = dict(old)
    for name in FILES:
        payload=(BASE/name).read_bytes()
        if name.endswith('.py'):
            ast.parse(payload)
        hashes[name]=hashlib.sha256(payload).hexdigest()
    output=BASE/'results/sft-phased-formal-source-manifest-20261025-01.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump({'schema':'jiuge7b-SFT-formal-continuation-sources-v1',
                   'utc':datetime.now(timezone.utc).isoformat(),'old169_unchanged':True,
                   'files':hashes,'only_new_orchestration_not_new_science':True},stream,indent=2)
    for name in FILES:
        subprocess.run([*SCP,str(BASE/name),'huids25@192.168.5.200:'+PREP+'/'+name],check=True,timeout=60)
    subprocess.run([*SCP,str(output),'huids25@192.168.5.200:'+PREP+'/'+output.name],check=True,timeout=60)
    code="from pathlib import Path; p=Path("+repr(PREP)+"); d=p/'sft-phased-formal-evidence-20261025-01'; d.mkdir(exist_ok=False); print(d)"
    subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,check=True,timeout=60)
    mapping={
      'results/sft-phased-correctness-driver-20261024-03/final-key-verification.json':'local-key-verification.json',
      'results/sft-phased-correctness-independent-review-20261024-03.json':'independent-review.json',
      'results/sft-phased-correctness-closure-20261024-03.json':'closure.json',
      'results/sft-phased-correctness-postprocess-source-20261024-03.json':'cpu-source-manifest.json'}
    for source,name in mapping.items():
        subprocess.run([*SCP,str(BASE/source),'huids25@192.168.5.200:'+PREP+'/sft-phased-formal-evidence-20261025-01/'+name],check=True,timeout=60)
    code='''from pathlib import Path
import ast,json,subprocess,sys
p=Path('''+repr(PREP)+''');sys.path.insert(0,str(p))
from sft_phased_formal_20261025 import configure,evidence_gate
c=configure();m=c.verify_sources();assert len(m)==179
admission=evidence_gate()
from run_pinning_formal import check_admission
# Recheck concrete original schemas/SHA using an ephemeral new admission fixture.
fixture=p/'sft-phased-formal-evidence-20261025-01/admission-CPU-fixture.json'
with fixture.open('x') as f:json.dump(admission,f,indent=2)
assert check_admission(fixture)['passed']
from continue_sft_phased_formal_20261025 import generated
text=generated();compile(text,'<formal-only-controller>','exec')
assert "case('phased-audit" not in text and "case('phased-native-checkpoint" not in text
assert "for seed,order in zip((63100,63101,63102)" in text
assert "if remaining() < 10800" in text
from review_sft_phased_formal_20261025 import generated as review
compile(review(),'<independent-same-timer-review>','exec')
import run_sft_phased_formal_20261025 as runner
original=(p/'run_native_case.py').read_text()
assert runner.derive_generated("parser.add_argument('--case',required=True)\\nstr(PREP/'sft_pinning_entry.py'),'--evidence'\\n    # This survives SIGKILL/OOM").count('sft_phased_acceptance_entry_20261024.py')==1
subprocess.run(['bash','-n',str(p/'bootstrap_sft_phased_formal_20261025.sh')],check=True)
means={s:{'native':4.0,'control':3.8,'optimized':3.6} for s in (63100,63101,63102)}
assert c.evaluate_incremental(means)['round_promotion_passed']
means[63100]['native']=3.0
assert not c.evaluate_incremental(means)['round_promotion_passed']
assert c.evaluate_incremental(means)['repeat_required']
print(json.dumps({'passed':True,'source_files':len(m),'old169_unchanged':True,
 'closed_correctness_revalidated':True,'full_raw_SHA_recomputed':False,
 'formal_derivations_compiled':True,'both_comparators_and_CV_negative_checks_passed':True,
 'GPU_used':False,'resource_submitted':False}))
'''
    result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=180)
    receipt={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
             'manifest_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'GPU_used':False}
    with (BASE/'results/sft-phased-formal-CPU-preparation-20261025-01.json').open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    print(result.stdout)


if __name__ == '__main__':
    main()
