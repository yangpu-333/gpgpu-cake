"""Low frequency host-memory observation, separate from training and GPU timing."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=7200.0)
    args = parser.parse_args()
    if args.seconds < 1 or args.timeout <= 0 or not args.case_root.is_dir():
        raise ValueError('Existing case and bounded low-frequency sampling required')
    os.nice(10)
    paths = [Path('/sys/fs/cgroup/memory.current'),
             Path('/sys/fs/cgroup/memory/memory.usage_in_bytes')]
    source = next((path for path in paths if path.is_file()), None)
    if source is None:
        raise RuntimeError('No supported container memory counter')
    started = time.monotonic()
    peak = samples = 0
    with (args.case_root / 'host-memory-samples.jsonl').open('x') as stream:
        while True:
            value = int(source.read_text())
            peak = max(peak, value)
            samples += 1
            stream.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(),
                                     'elapsed_seconds': time.monotonic() - started,
                                     'container_memory_bytes': value}) + '\n')
            stream.flush()
            if (args.case_root / 'case.json').exists() or time.monotonic() - started >= args.timeout:
                break
            time.sleep(args.seconds)
    receipt = {'schema': 'jiuge-host-memory-observation-v1', 'source': str(source),
               'sample_interval_seconds': args.seconds, 'sample_count': samples,
               'sampled_peak_bytes': peak, 'is_exact_kernel_peak': False,
               'gpu_api_calls': False, 'formal_performance_measurement': False}
    (args.case_root / 'host-memory-summary.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
