"""Official data + original reference; adapted runner, not official evaluator."""
import argparse
import functools
import hashlib
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from gdn_decode import vectorized, triton_candidate

NAME = 'gdn_decode_qk4_v8_d128_k_last'
ORDER = ('q','k','v','state','A_log','a','dt_bias','b','scale')


def load_candidate(path):
    spec = importlib.util.spec_from_file_location('kda_generated_candidate', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load candidate: {path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, 'triton_candidate', None)
    if not callable(fn):
        raise RuntimeError('candidate must export callable triton_candidate')
    return fn


def graph_for(fn, inputs, repetitions=10):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn(*inputs)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repetitions):
            outputs = fn(*inputs)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g, outputs


def measure(g, repetitions=10):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repetitions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--indices', default='', help='comma-separated official workload indices')
    p.add_argument('--candidate-source', help='isolated candidate module to validate')
    args = p.parse_args()
    data, output = Path(args.data), Path(args.output)
    if output.exists():
        raise SystemExit('Evidence output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    definition = json.loads((data/f'definitions/gdn/{NAME}.json').read_text())
    namespace = {}
    # This is the downloaded, reviewed official reference, not submission code.
    exec(compile(definition['reference'], 'official_reference', 'exec'), namespace)
    reference = namespace['run']
    workloads = [json.loads(line)['workload'] for line in (data/f'workloads/gdn/{NAME}.jsonl').read_text().splitlines()]
    if args.indices:
        indices = [int(value) for value in args.indices.split(',')]
        if any(index < 0 or index >= len(workloads) for index in indices):
            raise SystemExit('workload index out of range')
        workloads = [workloads[index] for index in indices]
    elif args.limit:
        workloads = workloads[:args.limit]
    functions = {'vectorized': vectorized}
    source_path = Path(__file__).with_name('gdn_decode.py')
    if args.candidate_source:
        source_path = Path(args.candidate_source).resolve()
        functions['candidate'] = load_candidate(source_path)
    else:
        for rows in (1, 4, 8, 16):
            functions[f'triton-r{rows}-w4'] = functools.partial(triton_candidate, rows=rows, warps=4)
    report = dict(scope='official inputs and reference; adapted V100 runner, not official FlashInfer acceptance',
                  dataset=json.loads((data/'dataset-lock.json').read_text()),
                  device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  atol=0.01, rtol=0.01, workloads=[], branches=[], status='running',
                  source_path=str(source_path),
                  source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest())
    try:
        for workload in workloads:
            spec = workload['inputs']
            blob_path = data / next(v['path'] for v in spec.values() if v['type']=='safetensors')
            tensors = load_file(str(blob_path), device='cuda')
            inputs = [tensors[spec[n]['tensor_key']] if spec[n]['type']=='safetensors' else spec[n]['value'] for n in ORDER]
            snapshots = [x.clone() if isinstance(x, torch.Tensor) else x for x in inputs]
            ref = reference(*inputs)
            item = dict(uuid=workload['uuid'], batch=inputs[0].shape[0], candidates={})
            valid = {}
            for name, fn in functions.items():
                try:
                    torch.cuda.reset_peak_memory_stats()
                    allocated_before = torch.cuda.memory_allocated()
                    got = fn(*inputs)
                    torch.cuda.synchronize()
                    peak_extra = torch.cuda.max_memory_allocated() - allocated_before
                    for g, r in zip(got, ref):
                        torch.testing.assert_close(g, r, atol=.01, rtol=.01)
                    for x, original in zip(inputs, snapshots):
                        if isinstance(x, torch.Tensor):
                            torch.testing.assert_close(x, original, atol=0, rtol=0)
                    errors = [float((g.float()-r.float()).abs().max()) for g,r in zip(got,ref)]
                    valid[name] = fn
                    item['candidates'][name] = dict(status='correct', max_abs=errors, peak_extra_bytes=peak_extra)
                except Exception as error:
                    item['candidates'][name] = dict(status='rejected', reason=str(error))
            if args.candidate_source and item['candidates']['candidate']['status'] != 'correct':
                report['workloads'].append(item)
                output.write_text(json.dumps(report,indent=2))
                raise RuntimeError(f"candidate rejected on {workload['uuid']}: {item['candidates']['candidate']['reason']}")
            if not valid:
                raise RuntimeError('No candidate passed')
            graphs = {n:graph_for(fn,inputs) for n,fn in valid.items()}
            samples = {n:[] for n in graphs}
            names = list(graphs)
            for repeat in range(7):
                for n in (names if repeat%2==0 else names[::-1]):
                    samples[n].append(measure(graphs[n][0]))
            for n in graphs:
                item['candidates'][n].update(raw_ms=samples[n], median_ms=statistics.median(samples[n]))
            item['winner'] = min(graphs,key=lambda n:statistics.median(samples[n]))
            report['workloads'].append(item)
            output.write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(uuid=item['uuid'],batch=item['batch'],winner=item['winner'])),flush=True)
            del graphs
        # Branches absent from real traces are tested separately.
        for state_value, scale_value in ((None,None),(None,0.0),(inputs[3],0.0)):
            branch_inputs = list(inputs)
            branch_inputs[3], branch_inputs[8] = state_value, scale_value
            expected = reference(*branch_inputs)
            for name, fn in functions.items():
                got = fn(*branch_inputs)
                for g,r in zip(got,expected):
                    torch.testing.assert_close(g,r,atol=.01,rtol=.01)
            report['branches'].append(dict(state_none=state_value is None, scale=scale_value, status='passed'))
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        output.write_text(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
