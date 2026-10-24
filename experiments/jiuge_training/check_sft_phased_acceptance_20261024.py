"""CPU-only guards for role wiring, exact derivation and unchanged promotion."""
import ast
import builtins
import gc
import json
from pathlib import Path
import tempfile
import weakref

from sft_phased_acceptance_20261024 import PREP, pinned_backend, evaluate_incremental


def main():
    for name, formal in (('pinning_entry.py', False), ('pinning_formal_entry.py', True)):
        text = pinned_backend((PREP/name).read_bytes(), formal)
        ast.parse(text)
        assert "name = 'candidate_hybrid_optimizer.py'" in text
        assert "else 'native_hybrid_optimizer.py'" not in text
        assert 'atol=' not in text
    import run_sft_pinning_case as runner
    from run_sft_phased_acceptance_20261024 import derive_generated
    captured = []
    def capture(text, name, mode):
        captured.append(derive_generated(text))
        raise StopIteration('generated only; never launch training')
    runner.compile = capture
    try:
        runner.main()
    except StopIteration:
        pass
    assert len(captured) == 1
    ast.parse(captured[0])
    assert "str(PREP/'sft_phased_acceptance_entry_20261024.py')" in captured[0]
    assert "--phased-plan" in captured[0]
    # A 0.5% candidate improvement or one losing seed must never pass promotion.
    means = {s: {'native': 4., 'control': 3.535, 'optimized': 3.52} for s in (63100,63101,63102)}
    assert not evaluate_incremental(means)['round_promotion_passed']
    for s in means:
        means[s]['control'] = 3.7
    assert evaluate_incremental(means)['round_promotion_passed']
    means[63102]['native'] = 3.4
    assert not evaluate_incremental(means)['round_promotion_passed']
    means[63102]['native'] = 6.
    assert evaluate_incremental(means)['repeat_required']
    import torch
    assert not torch.cuda.is_initialized()
    from sft_phased_acceptance_entry_20261024 import install_steps
    import sft_phased_placement_20261023 as phase
    import sft_local_moments_v4_20261023 as moments
    import sft_moment_memory_guard_20261021 as memory
    originals = (phase.activate, phase.install_physical_guard, moments.install, memory.install)
    calls = []
    phase.activate = lambda *a: calls.append('activate') or {'selected_cpu_nodes':[0]}
    def instrument(name):
        def install(*a):
            calls.append(name)
            return lambda: calls.append('undo-'+name)
        return install
    phase.install_physical_guard = instrument('physical')
    moments.install = instrument('moments')
    memory.install = instrument('cgroup')
    try:
        with tempfile.TemporaryDirectory() as scratch:
            for policy in ('baseline','affinity-control','local-moments'):
                calls.clear()
                class FakeHDO:
                    def step(self):
                        return 7
                original_step = FakeHDO.step
                state, finish = install_steps(torch, FakeHDO, Path(scratch)/'plan.json',
                    'unbound' if policy=='baseline' else 'gpu-local8', policy, Path(scratch), 0, False)
                opt = FakeHDO()
                assert opt.step() == opt.step() == 7 and calls.count('activate') == 1
                ref = weakref.ref(opt)
                del opt
                gc.collect()
                assert ref() is None and state['optimizer']() is None
                finish()
                assert FakeHDO.step is original_step
                (Path(scratch)/'phased-policy-rank-0.json').unlink()
                if policy=='local-moments':
                    assert calls == ['activate','cgroup','physical','moments',
                                     'undo-moments','undo-physical','undo-cgroup']
    finally:
        phase.activate, phase.install_physical_guard, moments.install, memory.install = originals
    result = {'passed': True, 'GPU_used': False,
              'generated_backend_and_runner_compile': True, 'all_incremental_arms_use_pinned': True,
              'both_baseline_and_control_promotion_threshold_enforced': True,
              'one_losing_seed_rejected': True, 'original_CV_repeat_preserved': True}
    result.update(first_step_dispatch_once=True, replaced_optimizer_released=True,
                  guards_removed_in_reverse_order=True,
                  fake_HDO_test_is_wiring_only_not_numeric_evidence=True)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
