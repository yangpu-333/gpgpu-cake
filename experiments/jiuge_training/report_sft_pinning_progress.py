"""Read only small shared-storage progress; no GPU or training-process queries."""
from datetime import datetime, timezone
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ROOT = PREP/'sft-pinning-validation-20261020-01'


def main():
    path = ROOT/'controller.json'
    controller = json.loads(path.read_text()) if path.is_file() else {}
    result = {'utc':datetime.now(timezone.utc).isoformat(),'status':controller.get('status','bootstrap'),
              'completed_cases':controller.get('completed_cases',[]),
              'formal_performance_started':controller.get('formal_performance_started',False),'numeric':[]}
    for folder in sorted((ROOT/'numeric').glob('*/rank-*')):
        journal = folder/'tensor-journal.jsonl'
        if not journal.is_file():
            continue
        with journal.open('rb') as stream:
            stream.seek(max(0,journal.stat().st_size-32768))
            lines = stream.read().splitlines()
        last = None
        for line in reversed(lines):
            try:
                last = json.loads(line)
                break
            except (json.JSONDecodeError,UnicodeDecodeError):
                continue
        record = {'arm':folder.parent.name,'rank':folder.name,'raw_pack_bytes':(folder/'raw.zlibpack').stat().st_size,
                  'last_closed_field':last.get('field') if last else None,'last_field_passed':last.get('passed') if last else None}
        completion = folder/'completion.json'
        if completion.is_file():
            record['completion'] = json.loads(completion.read_text())
        result['numeric'].append(record)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
