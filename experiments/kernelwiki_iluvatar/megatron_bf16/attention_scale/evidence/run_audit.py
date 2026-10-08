"""Fresh stage32 three-arm held-out training; unchanged all-master comparison."""
import argparse
from pathlib import Path
import common as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('wide', 'deep'), required=True)
    parser.add_argument('--version', default='003', choices=('003',))
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--suffix', default='complete')
    args = parser.parse_args()
    c.require(args.suffix.replace('-', '').isalnum(), 'simple immutable case suffix required')
    contract_path, contract, manifest, provenance = c.prepare(args.model, args.version, args.probe)
    suffix = args.model + '-' + args.version + '-' + args.suffix
    folders = {arm: c.BASE / (args.model + '-' + arm + '-audit-' + args.version + '-' + args.suffix)
               for arm in ('native', 'control', 'optimized')}
    outputs = {'controlled': c.BASE / ('comparison-' + suffix + '.json'),
               'native_optimized': c.BASE / ('comparison-native-optimized-' + suffix + '.json'),
               'native_control': c.BASE / ('comparison-native-control-' + suffix + '.json')}
    output = c.BASE / ('audit-summary-' + suffix + '.json')
    c.require(not any(path.exists() for path in [*folders.values(), *outputs.values(), output]),
              'refusing to overwrite immutable stage32 audit evidence')
    calls = contract['model']['num_layers'] * 3
    summary = {'schema': 'native-full-attention-stage32-three-way-audit-v1', 'status': 'running',
               'passed': False, 'version': args.version, 'model_name': args.model, 'suffix': args.suffix,
               'candidate_sha256': provenance['candidate_sha256'], 'contract_sha256': c.sha(contract_path),
               'wrapper_sha256': c.sha(c.WRAPPER), 'runner_sha256': c.sha(Path(__file__)),
               'controller_sources_sha256': c.sources(), 'provenance': provenance,
               'frozen_comparator_sha256': c.sha(c.BF / 'compare_training.py'),
               'audit_seed': c.AUDIT_SEEDS[args.model], 'comparisons': {}, 'completed_cases': [],
               'model': contract['model'], 'started_utc': c.utc()}
    try:
        hooks = {}
        for arm in ('native', 'control', 'optimized'):
            c.run_case(folders[arm], arm, args.model, args.version, contract_path, contract,
                       3, c.AUDIT_SEEDS[args.model])
            hooks[arm] = c.hook_gate(folders[arm], arm, args.version, contract, 3,
                                    manifest, provenance['backend'])
            summary['completed_cases'].append({'arm': arm, 'path': str(folders[arm]), 'hooks': hooks[arm]})
        import torch
        frozen = c.load_comparator()
        parent = c.BF / 'candidates/0008/candidate.py'

        class Controlled(frozen.Comparison):
            def require(self, condition, name, **details):
                if name == 'native has no candidate installation':
                    left = c.read(folders['control'] / 'entry.json')
                    condition = (left['iluvatar_kernels'] is True
                                 and len(left['kernel_adapters']) == 1
                                 and left['kernel_adapters'][0]['candidate_sha256'] == c.sha(parent)
                                 and all(hooks[arm]['gradnorm']['enabled'] is True
                                         and hooks[arm]['gradnorm']['candidate_sha256'] == c.sha(c.GRAD / 'candidate.py')
                                         and hooks[arm]['swiglu']['enabled'] is False
                                         and hooks[arm]['swiglu']['candidate_sha256'] is None
                                         and hooks[arm]['swiglu']['native_compiled_callable'] is True
                                         and hooks[arm]['swiglu']['native_calls'] == calls
                                         for arm in ('control', 'optimized'))
                                 and hooks['control']['attention']['candidate_calls'] == 0
                                 and hooks['optimized']['attention']['candidate_calls'] == calls
                                 and hooks['optimized']['attention']['candidate_sha256'] == c.CANDIDATE_SHA
                                 and hooks['optimized']['attention']['fallback_calls'] == 0)
                    name = 'Identical CE0008/gradnorm001 control and native SwiGLU; only Attention003 differs'
                    details = {'control_hooks': hooks['control'], 'optimized_hooks': hooks['optimized'],
                               'candidate_sha256': provenance['candidate_sha256']}
                elif name == 'native run mode':
                    command = c.read(folders['control'] / 'command.json')
                    condition = command['mode'] == 'optimized' and command['stage32_arm'] == 'control'
                    name = 'Control explicitly enables only frozen CE0008/gradnorm001'
                return super().require(condition, name, **details)

        comparisons = {'controlled': Controlled(folders['control'], folders['optimized'], torch,
                                                contract_path, parent),
                       'native_optimized': frozen.Comparison(folders['native'], folders['optimized'], torch,
                                                             contract_path, parent),
                       'native_control': frozen.Comparison(folders['native'], folders['control'], torch,
                                                           contract_path, parent)}
        for label, comparator in comparisons.items():
            report = comparator.run()
            report.update(stage32_comparison=label, model_name=args.model, version=args.version,
                          attention_candidate_sha256=provenance['candidate_sha256'],
                          stage32_runner_sha256=summary['runner_sha256'],
                          stage32_controller_sources_sha256=c.sources(),
                          stage32_probe_sha256=provenance['probe_sha256'])
            c.write(outputs[label], report)
            summary['comparisons'][label] = {
                'path': str(outputs[label]), 'sha256': c.sha(outputs[label]),
                'passed': report['passed'], 'check_summary': report['check_summary'],
                'tensor_comparisons': len(report['tensor_comparisons']),
                'all_tensors_exact_equal': all(item['exact_equal'] for item in report['tensor_comparisons']),
                'all_tensor_bytes_equal': all(item['native_sha256'] == item['optimized_sha256']
                                              for item in report['tensor_comparisons'])}
            print(c.json.dumps({'comparison': label, **summary['comparisons'][label]}), flush=True)
        summary.update(status='completed', passed=all(item['passed'] for item in summary['comparisons'].values()))
    except Exception as error:
        summary.update(status='rejected', passed=False, error_type=type(error).__name__, error=str(error))
    summary.update(finished_utc=c.utc(),
                   scope='Fresh native/control/optimized held-out three-step full-model training. '
                         'Original all600-source, model/input/loss/all-gradient/BF16-update/FP32-master '
                         'gates unchanged. Only two controlled-left candidate/run-mode assumptions '
                         'adapted. Native compiled SwiGLU and fused causal softmax in every arm. '
                         'Untimed native/control forward/backward route counters prove original APIs; '
                         'all-arm actual native SwiGLU observation and optimized four ScriptFunction '
                         'graphs retained. No performance, production, convergence or multiGPU claim.')
    c.write(output, summary)
    print(c.json.dumps({'passed': summary['passed'], 'output': str(output),
                        'error': summary.get('error')}), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
