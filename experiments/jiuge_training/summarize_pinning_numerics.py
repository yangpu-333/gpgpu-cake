"""Summarize completed full-value comparisons; raw integrity is a separate gate."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for arm in ('native', 'control', 'optimized'):
        for rank in (0, 1):
            folder = args.root / arm / f'rank-{rank}'
            payload = (folder / 'all-state-manifest.json').read_bytes()
            manifest = json.loads(payload)
            values = manifest['tensors']
            completion = json.loads((folder / 'completion.json').read_text())
            rows.append({'arm': arm, 'rank': rank, 'fields': len(values),
                         'failed_fields': sum(not r['passed'] for r in values),
                         'max_abs': max(r['max_abs'] for r in values),
                         'manifest_sha256': hashlib.sha256(payload).hexdigest(),
                         'completion': completion})
    result = {'schema': 'jiuge7b-pinning-numeric-summary-v1', 'rows': rows,
              'complete_numeric_comparisons_passed': all(r['fields'] == 3585 and
                  not r['failed_fields'] and r['completion']['passed'] for r in rows),
              'primary_raw_integrity_verification_included': False,
              'checkpoint_resume_verified': False, 'formal_performance_passed': False}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
