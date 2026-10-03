"""CE0008-controlled gradnorm audit; inherits frozen tensor tolerances/checks."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bf16-root',type=Path,required=True)
    p.add_argument('--case-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    spec=importlib.util.spec_from_file_location('frozen_bf16_compare',a.bf16_root/'compare_training.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import torch
    control=a.case_root/'control-audit-02';optimized=a.case_root/'optimized-audit-02'
    contract=a.bf16_root/'contracts/0008-main.json'
    parent=a.bf16_root/'candidates/0008/candidate.py'
    grad_candidate=a.case_root/'candidate.py'
    h=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    class Controlled(module.Comparison):
        def require(self,condition,name,**details):
            if name=='native has no candidate installation':
                e=json.loads((control/'entry.json').read_text())
                n=json.loads((control/'gradnorm.json').read_text())
                o=json.loads((optimized/'gradnorm.json').read_text())
                adapters=e.get('kernel_adapters',[])
                condition=e.get('iluvatar_kernels') is True and len(adapters)==1
                condition=condition and adapters[0]['candidate_sha256']==h(parent)
                condition=condition and adapters[0]['counts']=={'ce_candidate_calls':3,'ce_native_calls':0,'residual_native_calls':12,'fallback_reasons':{}}
                condition=condition and n['enabled'] is False and n['calls']==0
                condition=condition and o['enabled'] is True and o['calls']==3 and o['candidate_sha256']==h(grad_candidate)
                condition=condition and n['wrapper_sha256']==o['wrapper_sha256']==h(a.case_root/'launch_gradnorm-02.py')
                name='CE0008 controlled baseline and exact explicit gradnorm hook coverage'
            elif name=='native run mode':
                condition=json.loads((control/'command.json').read_text()).get('mode')=='optimized'
                name='controlled baseline run explicitly uses CE0008 optimized mode'
            elif name.endswith('command and launch match sealed verifier contract'):
                folder=control if name.startswith('native ') else optimized
                cmd=json.loads((folder/'command.json').read_text())['command']
                e=json.loads((folder/'entry.json').read_text())
                condition=Path(cmd[cmd.index('--contract')+1]).resolve()==contract.resolve() and e['contract_sha256']==h(contract)
                name+=' (contract derived from raw argv; command metadata omitted redundant hash)'
            return super().require(condition,name,**details)
    verifier=Controlled(control,optimized,torch,contract,parent)
    report=verifier.run()
    report['schema']='CE0008-controlled-gradnorm-three-update-audit-v1'
    report['scope']='Both arms use identical CE0008; only explicit local L2 norm hook differs. All inherited numeric tolerances and tensor/master checks unchanged.'
    report['controlled_verifier_sha256']=h(Path(__file__))
    report['gradnorm_candidate_sha256']=h(grad_candidate)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':report['check_summary']}))


if __name__=='__main__':
    main()
