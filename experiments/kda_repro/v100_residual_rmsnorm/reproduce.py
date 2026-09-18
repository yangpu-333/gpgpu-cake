"""Validate and compare forward CUDA graphs; preserve raw paired timings."""
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch
from src.reference import forward as baseline
from src.candidate import forward as candidate
from validate import validate_shape


def capture(fn, inputs):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(20):
            fn(*inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            outputs = fn(*inputs)
    return graph, outputs


def measure(graph):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        raise SystemExit('Refusing to overwrite evidence')
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260918)
    shapes = [(1, 33), (7, 127), (64, 768), (64, 1024), (64, 4096), (512, 4096)]
    report = dict(created=datetime.now(timezone.utc).isoformat(), device=torch.cuda.get_device_name(0),
                  torch=torch.__version__, cuda=torch.version.cuda, timing='forward CUDA graph replay; 9 alternating pairs, 100 forward calls inside each graph, one replay per sample', results=[])
    try:
        for shape in shapes:
            validate_shape(shape, candidate)
        report['validation'] = 'forward_and_reference_vjp_passed'
        if args.profile:
            x = torch.randn((64, 4096), device='cuda', dtype=torch.float16)
            candidate(x, x.clone(), torch.ones(4096, device='cuda', dtype=torch.float16))
            torch.cuda.synchronize()
        else:
            for shape in shapes:
                inputs = (torch.randn(shape, device='cuda', dtype=torch.float16), torch.randn(shape, device='cuda', dtype=torch.float16), torch.randn(shape[1], device='cuda', dtype=torch.float16))
                graphs = [capture(fn, inputs) for fn in (baseline, candidate)]
                samples = [[], []]
                for round_id in range(9):
                    for index in ((0, 1) if round_id % 2 == 0 else (1, 0)):
                        samples[index].append(measure(graphs[index][0]))
                medians = [statistics.median(s) for s in samples]
                report['results'].append(dict(shape=shape, baseline_ms=medians[0], candidate_ms=medians[1], speedup=medians[0]/medians[1], raw_ms=samples))
        report['status'] = 'validated_and_measured'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        target.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report))


if __name__ == '__main__':
    main()
