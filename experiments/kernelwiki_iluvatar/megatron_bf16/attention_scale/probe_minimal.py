"""Fresh-Pod allocation and execution check; no performance claim."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('immutable output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {'status': 'initializing', 'gpu_device': 0,
               'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'python': sys.version, 'machine': platform.machine(),
               'scope': 'Allocation, current-stream execution and CPU value check only; no timing benchmark.'}
    with open('/tmp/kda-biv150-gpu0.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            import torch
            receipt.update(torch=torch.__version__,
                           cuda_compatibility_label=torch.version.cuda,
                           cuda_available=torch.cuda.is_available(),
                           device_count=torch.cuda.device_count())
            if not receipt['cuda_available']:
                raise RuntimeError('GPU runtime unavailable')
            torch.cuda.set_device(0)
            prop = torch.cuda.get_device_properties(0)
            receipt.update(device_name=prop.name, total_memory_bytes=prop.total_memory,
                           compute_capability_compatibility_label=[prop.major, prop.minor])
            rows = []
            for dtype in (torch.float32, torch.bfloat16):
                x = torch.arange(256, device='cuda:0', dtype=dtype)
                y = x * 2 + 1
                torch.cuda.synchronize()
                actual = y.cpu().to(torch.float64)
                expected = x.cpu().to(torch.float64).mul(2).add(1).to(dtype).to(torch.float64)
                passed = bool(torch.equal(actual, expected))
                rows.append({'dtype': str(dtype), 'elements': x.numel(),
                             'allocation_completed': True, 'execution_completed': True,
                             'exact_cpu_value_check': passed})
                if not passed:
                    raise RuntimeError('minimal execution value check failed')
            receipt.update(status='passed', cases=rows,
                           compatibility_labels_are_not_nvidia_architecture_identity=True)
        except Exception as error:
            receipt.update(status='rejected', error_type=type(error).__name__, error=str(error))
        finally:
            receipt['finished_unix_seconds'] = time.time()
            with args.output.open('x', encoding='utf-8') as stream:
                json.dump(receipt, stream, indent=2, allow_nan=False)
                stream.write('\n')
    print(json.dumps(receipt, allow_nan=False), flush=True)
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
