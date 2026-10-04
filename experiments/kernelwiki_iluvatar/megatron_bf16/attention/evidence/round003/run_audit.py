"""Three fresh complete-attention training arms; unchanged all-master comparison."""
import argparse
from pathlib import Path
import common as c


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--suffix', default='complete')
    args = parser.parse_args()
    c.require(args.suffix.replace('-', '').isalnum(), 'simple immutable case suffix required')
    contract_path, contract, manifest, provenance = c.prepare(args.version, args.probe)
    suffix = 'main-' + args.version + '-' + args.suffix
    folders = {arm: c.BASE / ('main-' + arm + '-audit-' + args.version + '-' + args.suffix)
               for arm in ('native', 'control', 'optimized')}
    outputs = {'controlled': c.BASE / ('comparison-' + suffix + '.json'),
               'native_optimized': c.BASE / ('comparison-native-optimized-' + suffix + '.json'),
               'native_control': c.BASE / ('comparison-native-control-' + suffix + '.json')}
    output = c.BASE / ('audit-summary-' + suffix + '.json')
    c.require(not any(path.exists() for path in [*folders.values(), *outputs.values(), output]),
              'refusing to overwrite immutable Attention audit evidence')
    summary = {'schema': 'native-full-attention-stage31-three-way-audit-v1', 'status': 'running',
               'passed': False, 'version': args.version, 'model_name': 'main', 'suffix': args.suffix,
               'candidate_sha256': provenance['candidate_sha256'], 'contract_sha256': c.sha(contract_path),
               'wrapper_sha256': c.sha(c.WRAPPER), 'runner_sha256': c.sha(Path(__file__)),
               'controller_sources_sha256': c.sources(), 'provenance': provenance,
               'frozen_comparator_sha256': c.sha(c.BF / 'compare_training.py'),
               'audit_seed': c.AUDIT_SEED, 'comparisons': {}, 'completed_cases': [], 'started_utc': c.utc()}
    try:
        hooks = {}
        for arm in ('native', 'control', 'optimized'):
            c.run_case(folders[arm], arm, args.version, contract_path, contract, 3, c.AUDIT_SEED)
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
                                 and hooks['control']['swiglu']['enabled'] is True
                                 and hooks['optimized']['swiglu']['enabled'] is True
                                 and hooks['control']['swiglu']['candidate_sha256'] == c.SWIGLU_SHA
                                 and hooks['optimized']['swiglu']['candidate_sha256'] == c.SWIGLU_SHA
                                 and hooks['control']['attention']['candidate_calls'] == 0
                                 and hooks['optimized']['attention']['candidate_calls'] == 12
                                 and hooks['optimized']['attention']['fallback_calls'] == 0)
                    name = 'Identical CE0008/gradnorm001/SwiGLU004 control; only complete Attention differs'
                    details = {'control_hooks': hooks['control'], 'optimized_hooks': hooks['optimized'],
                               'candidate_sha256': provenance['candidate_sha256']}
                elif name == 'native run mode':
                    command = c.read(folders['control'] / 'command.json')
                    condition = command['mode'] == 'optimized' and command['stage31_arm'] == 'control'
                    name = 'Control explicitly enables only frozen CE0008/gradnorm001/SwiGLU004'
                return super().require(condition, name, **details)

        comparisons = {'controlled': Controlled(folders['control'], folders['optimized'], torch,
                                                contract_path, parent),
                       'native_optimized': frozen.Comparison(folders['native'], folders['optimized'], torch,
                                                             contract_path, parent),
                       'native_control': frozen.Comparison(folders['native'], folders['control'], torch,
                                                           contract_path, parent)}
        for label, comparator in comparisons.items():
            report = comparator.run()
            report.update(stage31_comparison=label, model_name='main', version=args.version,
                          attention_candidate_sha256=provenance['candidate_sha256'],
                          stage31_runner_sha256=summary['runner_sha256'],
                          stage31_controller_sources_sha256=c.sources(),
                          stage31_probe_sha256=provenance['probe_sha256'])
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
                   scope='Fresh native/control/optimized three-step full-model runs. Original 600-source, '
                         'model/input/loss/all-gradient/BF16-update/FP32-master gates unchanged. '
                         'Only controlled-left candidate/run-mode identity assumptions adapted. '
                         'All arms retain default native fused softmax flags; native/control route counters '
                         'prove original fusion APIs forward/backward, without a GPU timeline claim. '
                         'Native RMSNorm; no performance, heldout, production or convergence claim.')
    c.write(output, summary)
    print(c.json.dumps({'passed': summary['passed'], 'output': str(output),
                        'error': summary.get('error')}), flush=True)
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
