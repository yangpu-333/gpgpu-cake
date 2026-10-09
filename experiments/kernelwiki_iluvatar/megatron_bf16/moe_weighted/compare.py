"""Fresh MoE all-model/master comparison using the unchanged BF16 tensor reader."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
BF = Path('/tmp/kda-bf16-continuation/megatron_bf16')
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())


def check_hook(folder, arm):
    r, g = read(folder / 'weighted.json'), read(folder / 'gradnorm.json')
    assert r['wrapper_sha256'] == sha(ROOT / 'launch_weighted.py') and r['calls'] == 48
    assert r['enabled'] is (arm == 'optimized') and r['candidate_calls'] == (48 if arm == 'optimized' else 0)
    assert r['native_calls'] == (0 if arm == 'optimized' else 48) and r['fallback_calls'] == 0
    assert r['sort_compat_calls'] == r['router_calls'] == len(r['expert_calls']) == 12
    assert all(sum(row['tokens_per_expert']) == 512 and row['class'] == 'SequentialMLP'
               for row in r['expert_calls'])
    assert r['native_compiled_callables'] == {'weighted_swiglu': True, 'weighted_swiglu_back': True}
    assert r['configurations'] == [{'num_moe_experts': 4, 'moe_router_topk': 2,
        'moe_grouped_gemm': False, 'moe_token_dispatcher_type': 'allgather',
        'moe_apply_probs_on_input': False, 'moe_aux_loss_coeff': .01,
        'gated_linear_unit': True, 'bias_activation_fusion': True,
        'masked_softmax_fusion': True, 'use_te_activation_func': False}]
    assert g['enabled'] is (arm != 'native') and g['calls'] == (3 if arm != 'native' else 0)
    assert g['fallback_calls'] == 0 and (not g['norm_audit'] if arm == 'native'
           else len(g['norm_audit']) == 3 and all(row['bitwise_equal'] for row in g['norm_audit']))
    return r, g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--left-arm', choices=('native', 'control'), required=True)
    p.add_argument('--right-arm', choices=('control', 'optimized'), required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    output = ROOT / a.output; assert not output.exists()
    path = BF / 'compare_training.py'
    assert sha(path) == 'feb475eb504fe452267f930f2069d5f17b36387a6cb4ad0f839a6aaa8df27eda'
    spec = importlib.util.spec_from_file_location('frozen_bf16_moe_compare', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    import torch
    torch.set_num_threads(4)
    left, right = ROOT / a.left, ROOT / a.right
    hooks = {arm: check_hook(folder, arm) for arm, folder in ((a.left_arm, left), (a.right_arm, right))}
    class Controlled(m.Comparison):
        def require(self, condition, name, **details):
            if a.left_arm == 'control' and name == 'native run mode':
                record = read(left / 'command.json')
                condition = record['mode'] == 'optimized' and record['stage34_arm'] == 'control'
                name = 'Control explicitly installs only frozen CE0008/gradnorm001'
            elif a.left_arm == 'control' and name == 'native has no candidate installation':
                entry = read(left / 'entry.json')
                condition = (entry['iluvatar_kernels'] is True and len(entry['kernel_adapters']) == 1
                    and entry['kernel_adapters'][0]['candidate_sha256'] == sha(BF / 'candidates/0008/candidate.py')
                    and hooks['control'][0]['enabled'] is False and hooks[a.right_arm][0]['enabled'] is True
                    and all(hook[1]['enabled'] is True for hook in hooks.values()))
                name = 'Same frozen CE/gradnorm control; only weighted SwiGLU candidate differs'
            return super().require(condition, name, **details)
    comparison = Controlled(left, right, torch, ROOT / 'contract.json', BF / 'candidates/0008/candidate.py')
    value = comparison.run()
    # Extend the numerical scope with actual router outputs/maps and probability gradients.
    routes = []
    for number in range(1, 13):
        for name in ('router-%03d.pt' % number, 'probability-grad-%03d.pt' % number):
            paths = [folder / 'routing' / name for folder in (left, right)]
            records = [torch.load(path, map_location='cpu', weights_only=True) for path in paths]
            comparison.compare_tree(*records, 'routing/' + name, 'routing', exact='router-' in name)
            routes.extend({'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)} for path in paths)
    value.update(checks=comparison.checks, tensor_comparisons=comparison.tensors,
        check_summary={'total':len(comparison.checks), 'passed':sum(row['passed'] for row in comparison.checks),
                       'failed':sum(not row['passed'] for row in comparison.checks)},
        passed=all(row['passed'] for row in comparison.checks), stage34_comparator_sha256=sha(__file__),
        route_input_files=routes, left_arm=a.left_arm, right_arm=a.right_arm,
        scope='Frozen original all-loss/all-gradient/BF16-parameter/FP32-master gates, plus original '
              'router probabilities/maps and probability gradients. Only two controlled-left metadata assertions adapted.')
    output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'passed':value['passed'],'checks':value['check_summary'],
                      'tensors':len(value['tensor_comparisons']), 'failed_examples':[r['name'] for r in comparison.checks if not r['passed']][:8]}))
    return 0 if value['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
