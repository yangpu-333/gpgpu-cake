"""Read only completed journal rows; no tensor bodies, CUDA calls or mutations."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    root = PREP / 'pinning-validation-20261015-04'
    controller = json.loads((root / 'controller.json').read_text())
    status = controller['status']
    arm = {'waiting_for_native': 'native', 'running_control': 'control', 'running_optimized': 'optimized'}.get(status)
    result = {'time_cst': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'campaign_status': status, 'active_arm': arm,
              'completed_arms': [row['arm'] for row in controller.get('runs', [])], 'ranks': [],
              'formal_performance_started': False}
    if arm:
        for rank in (0, 1):
            path = root / arm / f'rank-{rank}/tensor-journal.jsonl'
            # A final row still being appended is not counted or decoded.
            lines = path.read_bytes().split(b'\n')[:-1] if path.is_file() else []
            rows = [json.loads(line) for line in lines if line]
            result['ranks'].append({'rank': rank, 'compared_fields': len(rows), 'expected_fields': 3585,
                                    'failed_fields': sum(not row['passed'] for row in rows),
                                    'max_abs': max((row['max_abs'] for row in rows), default=0.0),
                                    'latest_field': rows[-1]['field'] if rows else None})
    if 'error' in controller:
        result['error'] = controller['error']
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
