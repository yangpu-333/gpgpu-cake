"""Two-rank GPU/collective tracing capability probe; NOT Jiuge training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import time


def main():
    import torch
    import torch.distributed as distributed
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    if world != 2:
        raise ValueError('This capability probe requires exactly two ranks')
    folder = args.output / ('rank-' + str(rank))
    folder.mkdir(parents=True, exist_ok=False)
    torch.cuda.set_device(local_rank)
    supported = torch.profiler.supported_activities()
    if torch.profiler.ProfilerActivity.CUDA not in supported:
        raise RuntimeError('Device profiler activity is not supported; CPU-only is insufficient')
    distributed.init_process_group('nccl')
    torch.manual_seed(32100 + rank)
    matrix = torch.randn((1024, 1024), dtype=torch.bfloat16, device='cuda')
    collective = torch.full((1024 * 1024,), rank + 1., device='cuda')
    for _ in range(5):
        distributed.all_reduce(collective)
        collective.fill_(rank + 1.)
        matrix @ matrix
    distributed.barrier()
    torch.cuda.synchronize()
    window_start_ns = time.time_ns()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
        with torch.profiler.record_function('toolchain_probe/full_window'):
            for step in range(5):
                with torch.profiler.record_function('toolchain_probe/step/' + str(step)):
                    collective.fill_(rank + 1.)
                    with torch.profiler.record_function('toolchain_probe/async_all_reduce'):
                        request = distributed.all_reduce(collective, async_op=True)
                    with torch.profiler.record_function('toolchain_probe/matmul'):
                        product = matrix @ matrix
                    request.wait()
            # Only the window boundary is drained. Internal async overlap remains.
            torch.cuda.synchronize()
    window_end_ns = time.time_ns()
    trace = folder / 'trace.json'
    profiler.export_chrome_trace(str(trace))
    value = json.loads(trace.read_text())
    kernels = [event for event in value['traceEvents'] if event.get('cat') == 'kernel']
    communications = [event for event in kernels if 'nccl' in event.get('name', '').lower()]
    expected = torch.full_like(collective, 3.)
    correct = torch.equal(collective, expected) and bool(torch.isfinite(product).all().item())
    record = {'schema': 'two-rank-profiler-capability-probe-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'hostname': socket.gethostname(),
              'rank': rank, 'local_rank': local_rank, 'world_size': world,
              'gpu_name': torch.cuda.get_device_name(local_rank),
              'torch': torch.__version__, 'nccl': torch.cuda.nccl.version(),
              'window_start_wall_ns': window_start_ns, 'window_end_wall_ns': window_end_ns,
              'trace_base_time_nanoseconds': value.get('baseTimeNanoseconds'),
              'gpu_kernel_events': len(kernels), 'gpu_nccl_kernel_events': len(communications),
              'collective_numerical_probe_passed': correct,
              'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
              'workload': 'small matmul and async all-reduce capability probe',
              'uses_real_jiuge_model': False, 'end_to_end_jiuge_tracing_completed': False,
              'formal_throughput_measurement': False,
              'passed': correct and bool(kernels) and bool(communications)}
    (folder / 'receipt.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    distributed.destroy_process_group()
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
