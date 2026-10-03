"""Freeze delivery receipts after successful formal training validation."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    evidence = ROOT/'evidence'
    checks = {}
    for name in ('audit-comparison-01.json','resume-comparison-01.json','performance-120-validated.json'):
        report = json.loads((evidence/name).read_text())
        checks[name] = {'passed':report['passed'],'sha256':sha(evidence/name),
                        'check_summary':report.get('check_summary')}
    dispatch = json.loads((evidence/'formal-dispatch-01/dispatch.json').read_text())
    expected = ('_forward_kernel','_backward_rows_kernel','_backward_weight_kernel',
                '_ce_forward_kernel','_ce_backward_kernel')
    checks['compiled_dispatch'] = {'passed':dispatch['status']=='completed' and all(
        any(name.startswith(prefix) and count>0 for name,count in dispatch['counts'].items()) for prefix in expected),
        'counts':dispatch['counts'], 'scope':dispatch['scope'],
        'sha256':sha(evidence/'formal-dispatch-01/dispatch.json')}
    for label, command in (
        ('adapter-cpu-tests',[sys.executable,'-m','unittest','discover','-s',str(ROOT/'tests'),'-v']),
        ('performance-parser-cpu-tests',[sys.executable,str(ROOT/'summarize_performance.py'),'--self-test'])):
        result = subprocess.run(command,capture_output=True)
        (evidence/(label+'.stdout')).write_bytes(result.stdout)
        (evidence/(label+'.stderr')).write_bytes(result.stderr)
        checks[label] = {'passed':result.returncode==0,'exit_code':result.returncode}
    candidate = ROOT.parent/'megatron_cc/higher_gain/candidates/0006/candidate.py'
    checks['candidate_unchanged'] = {'passed':sha(candidate)==
        '4a5c2180114396412b1c99907837ab970b2a9efdffe85e0a46f5527cc0790201', 'sha256':sha(candidate)}
    # Check actual configured keys without printing their values.
    secrets = [value.encode() for name,value in os.environ.items()
               if value and len(value)>12 and ('API_KEY' in name or 'AUTH_TOKEN' in name)]
    published = [path for path in ROOT.rglob('*') if path.is_file()
                 and 'snapshots' not in path.relative_to(ROOT).parts
                 and '__pycache__' not in path.relative_to(ROOT).parts
                 and path.suffix != '.gz']
    leaked = [str(path.relative_to(ROOT)) for path in published
              if any(secret in path.read_bytes() for secret in secrets)]
    checks['configured_secret_scan'] = {'passed':not leaked,'configured_values_checked':len(secrets),
                                        'matching_files':leaked}
    perf = json.loads((evidence/'performance-120-validated.json').read_text())
    report = {'schema':'megatron-formal-entry-delivery-v1','finished_utc':datetime.now(timezone.utc).isoformat(),
        'passed':all(check['passed'] for check in checks.values()), 'checks':checks,
        'formal_training_throughput_ratio':perf['geometric_mean_native_over_optimized_ratio'],
        'scope':'Four-layer FP32, native MockGPTDataset, TP/PP/CP1. No production, BF16 fusion, SFT or multi-GPU claim.'}
    output = evidence/'delivery-decision.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2); stream.write('\n')
    manifest = [{'path':path.relative_to(ROOT).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)}
                for path in sorted(set(published+[output]))]
    with (evidence/'delivery-manifest.json').open('x',encoding='utf-8') as stream:
        json.dump({'files':manifest,'count':len(manifest),'excluded':'snapshots, __pycache__, transport archives, manifest itself'},stream,indent=2)
        stream.write('\n')
    print(json.dumps({'passed':report['passed'],'files':len(manifest),'output':str(output)}))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
