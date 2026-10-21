"""CPU-only regression for real topology hazards and exact runner derivation."""
import hashlib
import json
from pathlib import Path
from sft_numa_placement_20261021 import build_plan
from run_sft_numa_case_20261021 import derive

BASE = Path(__file__).resolve().parent


def topology(gpu_nodes=(1, 1), physical_per_node=16):
    cpus = {}
    # Deliberately include SMT siblings; a valid plan must never count them twice.
    for node in (0, 1):
        for core in range(physical_per_node):
            for sibling in (0, 1):
                cpu = node*physical_per_node + core + sibling*physical_per_node*2
                cpus[str(cpu)] = {'node': node, 'package': str(node), 'core': str(core)}
    return {'allowed_cpus': sorted(map(int, cpus)), 'cpu_topology': cpus,
            'gpus': [{'uuid': f'fixture-gpu-{r}', 'pci': f'0000:{r+1:02x}:00.0', 'node': n}
                     for r, n in enumerate(gpu_nodes)]}


def main():
    checks = []
    shared = topology()
    plan = build_plan(shared)
    assert set(plan['policies']) == {'unbound', 'gpu-local8', 'socket-split8'}
    for mode, ranks in plan['policies'].items():
        if mode == 'unbound':
            assert ranks == [shared['allowed_cpus'], shared['allowed_cpus']]
            continue
        used = set()
        for selected in ranks:
            physical = {(shared['cpu_topology'][str(c)]['package'],
                         shared['cpu_topology'][str(c)]['core']) for c in selected}
            assert len(selected) == len(physical) == 8 and not used.intersection(physical)
            used.update(physical)
    assert all(shared['cpu_topology'][str(c)]['node'] == 1
               for rank in plan['policies']['gpu-local8'] for c in rank)
    assert [{shared['cpu_topology'][str(c)]['node'] for c in rank}
            for rank in plan['policies']['socket-split8']] == [{0}, {1}]
    checks.append('Both GPUs on NUMA1: no rank-to-node assumption; disjoint physical cores; split-node alternative')
    separate = build_plan(topology((0, 1)))
    assert set(separate['policies']) == {'unbound', 'gpu-local8'}
    checks.append('Different GPU nodes: duplicate split policy omitted rather than measured as another candidate')
    try:
        build_plan(topology(physical_per_node=8))
    except ValueError:
        checks.append('Sixteen SMT threads on eight physical cores cannot serve two local eight-core ranks')
    else:
        raise AssertionError('Insufficient physical cores accepted')
    bad = topology((-1, 1))
    try:
        build_plan(bad)
    except ValueError:
        checks.append('Unknown GPU NUMA node rejected instead of guessed')
    else:
        raise AssertionError('Unknown GPU node accepted')

    # Generate the exact native launch source without importing Torch, touching
    # the GPU or executing training. Check the resulting program, not just the
    # outer adapter's syntax (which previously hid runner derivation failures).
    source = (BASE/'run_sft_trace_case_20261020.py').read_bytes()
    namespace = {'__name__': 'probe', '__file__': str(BASE/'run_sft_numa_case_20261021.py')}
    exec(compile(derive(source), '<SFT NUMA wrapper>', 'exec'), namespace)
    namespace['PREP'] = BASE
    generated = []
    namespace['compile'] = lambda text, *args: text
    namespace['exec'] = lambda text, *args: generated.append(text)
    namespace['generated_main']()
    assert len(generated) == 1
    inner = generated[0]
    compile(inner, '<derived native runner>', 'exec')
    assert "str(PREP/'sft_numa_entry_20261021.py')" in inner
    assert "choices=('optimized',)" in inner
    assert "datasets/alpaca-sft-answer-v1" in inner
    assert "'--sft-numa-mode',args.sft_numa_mode" in inner
    assert "'--sft-numa-plan',str(args.sft_numa_plan)" in inner
    assert "OMP_NUM_THREADS='8'" in inner
    assert 'Fixed 35-step S2048 profile only' in inner
    checks.append('Exact frozen SFT-to-native derivation compiles, keeps pinned repair, SFT data, S2048,35 steps and OMP8')
    print(json.dumps({'schema': 'jiuge7b-sft-numa-CPU-preparation-probe-v1',
                      'passed': True, 'checks': checks,
                      'frozen_sft_runner_sha256': hashlib.sha256(source).hexdigest(),
                      'torch_imported': False, 'GPU_used': False,
                      'real_training_numeric_or_performance_verified': False}, indent=2))


if __name__ == '__main__':
    main()
