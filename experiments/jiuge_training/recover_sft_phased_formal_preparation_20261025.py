"""Complete a missing CPU-review template dependency; no science or GPU change."""
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from drive_sft_pinning_validation import PREP, SSH, SCP

BASE=Path(__file__).resolve().parent
if __name__=='__main__':
    template=BASE/'review_sft_phased_acceptance_20261024.py'
    supplemental={template.name:hashlib.sha256(template.read_bytes()).hexdigest()}
    receipt=BASE/'results/sft-phased-formal-postprocess-sources-20261025-01.json'
    record={'files':supplemental,'CPU_postprocessing_only':True,'science179_unchanged':True}
    if receipt.exists():
        assert json.loads(receipt.read_text())==record
    else:
        with receipt.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2)
    subprocess.run([*SCP,str(template),'huids25@192.168.5.200:'+PREP+'/'+template.name],check=True,timeout=60)
    subprocess.run([*SCP,str(receipt),'huids25@192.168.5.200:'+PREP+'/'+receipt.name],check=True,timeout=60)
    text=(BASE/'prepare_sft_phased_formal_20261025.py').read_text()
    start=text.index("    code='''from pathlib import Path")
    stop=text.index('    result=subprocess.run',start)
    scope={'PREP':PREP}
    exec(text[start:stop].lstrip(),scope)
    code=scope['code'].replace('admission-CPU-fixture.json','admission-CPU-fixture-v3.json')
    code=code.replace('runner.derive_generated(', 'runner.generated(')
    code=code.replace(".count('sft_phased_acceptance_entry_20261024.py')==1", ".count('sft_phased_formal_entry_20261025.py')==1")
    result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,timeout=180)
    record={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr,
            'old169_and_new179_unchanged':True,'missing_CPU_template_supplied':supplemental,'GPU_used':False}
    with (BASE/'results/sft-phased-formal-CPU-preparation-20261025-03.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2)
    if result.returncode:raise RuntimeError(result.stderr)
    print(result.stdout)
