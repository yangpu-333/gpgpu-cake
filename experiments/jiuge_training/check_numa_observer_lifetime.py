"""Exercise actual observer hooks under optimizer replacement, without Torch/GPU."""
import gc
import json
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import weakref

BASE = Path(__file__).resolve().parent


def run_entry(name, dummy_update):
    namespace = {'__name__': 'observer_lifetime_fixture', '__file__': str(BASE / name)}
    exec(compile((BASE / name).read_text(), str(BASE / name), 'exec'), namespace)
    fixtures = BASE / 'results/lifetime-fixtures'
    fixtures.mkdir(exist_ok=True)
    assert fixtures.resolve().is_relative_to(BASE.resolve())
    with tempfile.TemporaryDirectory(prefix='numa-lifetime-', dir=fixtures) as temporary:
        root = Path(temporary)
        assert root.resolve().is_relative_to(fixtures.resolve())
        for source in ('numa_entry.py', 'numa_observation_v2_entry.py', 'numa_placement.py'):
            (root / source).write_bytes((BASE / source).read_bytes())
        namespace['PREP'] = root
        namespace['bind'] = lambda mode, rank: {'mode': mode, 'rank': rank, 'fixture_only': True}
        namespace['observe'] = lambda opt, phase, torch: {'phase': phase, 'fixture_only': True}

        class HDO:
            def __init__(self):
                self._init_sub_optimizers()

            def _init_sub_optimizers(self):
                return None

            def step(self):
                return 1

        modules = {'torch': types.ModuleType('torch')}
        names = ('megatron', 'megatron.core', 'megatron.core.optimizer',
                 'megatron.core.optimizer.cpu_offloading',
                 'megatron.core.optimizer.cpu_offloading.hybrid_optimizer')
        for module_name in names:
            module = types.ModuleType(module_name)
            module.__path__ = []
            modules[module_name] = module
            if '.' in module_name:
                parent, child = module_name.rsplit('.', 1)
                setattr(modules[parent], child, module)
        modules[names[-1]].HybridDeviceOptimizer = HDO
        held, result = {}, {}

        def replaced_optimizer_backend(*args, **kwargs):
            initial = HDO()
            if dummy_update:
                initial.step()
            previous = weakref.ref(initial)
            held['current'] = HDO()
            del initial
            gc.collect()
            result['replaced_optimizer_collected_before_real_update'] = previous() is None
            held['current'].step()
            result['active_reference'] = weakref.ref(held['current'])

        old_path = sys.path[:]
        try:
            with patch.dict(sys.modules, modules), patch.dict(os.environ, {'RANK': '0'}), \
                    patch.object(sys, 'argv', [name, '--numa-mode', 'local8', '--numa-root', str(root),
                                              '--numa-observe', '--numa-backend', 'pinning_entry.py']), \
                    patch.object(namespace['runpy'], 'run_path', replaced_optimizer_backend):
                namespace['main']()
        finally:
            sys.path[:] = old_path
        held.clear()
        gc.collect()
        result['active_optimizer_collected_after_model_release'] = result.pop('active_reference')() is None
        return {'entry': name, 'initialization_dummy_update': dummy_update, **result}


def main():
    rows = [run_entry(name, dummy) for name in ('numa_entry.py', 'numa_observation_v2_entry.py')
            for dummy in (False, True)]
    for row in rows:
        fixed = row['entry'] == 'numa_observation_v2_entry.py'
        assert row['replaced_optimizer_collected_before_real_update'] == fixed
        assert row['active_optimizer_collected_after_model_release'] == fixed
    import hashlib
    proof = {'schema': 'jiuge7b-observer-lifetime-regression-v1', 'passed': True, 'rows': rows,
             'source_sha256': {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in
                               ('numa_entry.py', 'numa_observation_v2_entry.py', 'numa_placement.py', 'check_numa_observer_lifetime.py')},
             'scope': 'Actual entry hooks with simulated optimizer replacement; no Torch/CUDA, numeric or performance proof.'}
    (BASE / 'results/cpu-numa-observer-lifetime-regression-20261016.json').write_text(
        json.dumps(proof, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
