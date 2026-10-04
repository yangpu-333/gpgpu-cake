"""Heldout native synchronized training throughput for frozen SwiGLU004."""
import argparse
from datetime import datetime
import math
from pathlib import Path
import statistics
import common as c


def numerical_gate(model, suffix, contract_path, contract, manifest, adaptation):
    path = c.BASE / ('audit-summary-' + model + '-004-' + suffix + '.json')
    audit = c.read(path)
    c.require(audit['schema'] == 'fixed-swiglu-stage30-three-way-audit-v1'
              and audit['status'] == 'completed' and audit['passed'] is True
              and audit['model_name'] == model and audit['suffix'] == suffix
              and audit['audit_seed'] == c.AUDIT_SEEDS[model]
              and audit['candidate_sha256'] == c.CANDIDATE_SHA
              and audit['contract_sha256'] == c.sha(contract_path)
              and audit['wrapper_sha256'] == c.sha(c.WRAPPER)
              and audit['common_sha256'] == c.sha(c.BASE / 'common.py')
              and audit['runner_sha256'] == c.sha(c.BASE / 'run_audit.py')
              and audit['adaptation_record_sha256'] == c.sha(c.BASE / 'adaptation-record.json')
              and audit['frozen_sources_sha256'] == adaptation['frozen_sources_sha256']
              and audit['frozen_comparator_sha256'] == c.sha(c.BF / 'compare_training.py'),
              'unaltered passing same-model three-way numerical audit required')
    c.require(set(audit['comparisons']) == {'controlled', 'native_optimized', 'native_control'}
              and len(audit['completed_cases']) == 3, 'complete three-arm heldout audit required')
    inventories = {}
    for label, item in audit['comparisons'].items():
        comparison = Path(item['path'])
        proof = c.read(comparison)
        c.require(comparison.resolve().parent == c.BASE.resolve() and c.sha(comparison) == item['sha256']
                  and item['passed'] is True and proof['passed'] is True
                  and proof['stage30_comparison'] == label and proof['model_name'] == model
                  and proof['steps'] == 3 and proof['atol'] == 3e-4 and proof['rtol'] == 1e-3
                  and proof['check_summary']['failed'] == 0
                  and all(check['passed'] for check in proof['checks'])
                  and all(tensor['passed'] for tensor in proof['tensor_comparisons'])
                  and proof['comparator_sha256'] == audit['frozen_comparator_sha256']
                  and proof['stage30_runner_sha256'] == audit['runner_sha256']
                  and proof['swiglu_candidate_sha256'] == c.CANDIDATE_SHA,
                  'complete frozen numerical comparison differs: ' + label)
        computed = {'total': len(proof['checks']),
                    'passed': sum(check['passed'] is True for check in proof['checks']),
                    'failed': sum(check['passed'] is not True for check in proof['checks'])}
        c.require(proof['check_summary'] == computed and computed['total'] > 0
                  and bool(proof['tensor_comparisons'])
                  and item['check_summary'] == computed
                  and item['tensor_comparisons'] == len(proof['tensor_comparisons']),
                  'numerical comparison summary does not reflect all raw gate/tensor records')
        for row in proof['input_files']:
            raw = Path(row['path']).resolve()
            c.require(any(raw.is_relative_to(root.resolve()) for root in
                          (c.BASE, c.BF, c.FORMAL, c.GRAD, c.SEALED)),
                      'numerical input inventory escaped the fixed experiment/source roots')
            previous = inventories.setdefault(str(raw), row)
            c.require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                      'same inventoried numerical input has conflicting metadata')
    for raw_name, row in inventories.items():
        raw = Path(raw_name)
        c.require(raw.is_file() and raw.stat().st_size == row['bytes']
                  and c.sha(raw) == row['sha256'],
                  'raw tensor/metadata bytes changed after full numerical audit: ' + raw_name)
    for item in audit['completed_cases']:
        folder = Path(item['path'])
        c.require(folder.resolve().parent == c.BASE.resolve(), 'unexpected audited case path')
        current = c.hook_gate(folder, item['arm'], contract, 3, manifest)
        c.require(item['hooks'] == current, 'raw numerical runtime receipts differ from sealed audit')
    return path, len(inventories)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('wide', 'deep'), required=True)
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    args = parser.parse_args()
    c.require(all(value.replace('-', '').isalnum() for value in (args.audit_suffix, args.run_suffix)),
              'simple immutable audit/run suffixes required')
    contract_path, contract, manifest, adaptation = c.prepare(args.model)
    audit_path, verified_inputs = numerical_gate(
        args.model, args.audit_suffix, contract_path, contract, manifest, adaptation)
    seeds = contract['performance']['independent_seeds']
    suffix = args.model + '-004-' + args.run_suffix
    output = c.BASE / ('performance-' + suffix + '.json')
    folders = {(seed, arm): c.BASE / (args.model + '-' + arm + '-perf-004-' + args.run_suffix + '-' + str(seed))
               for seed in seeds for arm in ('native', 'control', 'optimized')}
    c.require(not any(path.exists() for path in [output, *folders.values()]),
              'refusing to overwrite any immutable heldout performance evidence')
    validator, validator_adaptation = c.load_performance_validator()
    report = {'schema': 'fixed-swiglu-stage30-three-arm-performance-v1', 'status': 'running',
              'promoted': False, 'model_name': args.model, 'run_suffix': args.run_suffix,
              'candidate_sha256': c.CANDIDATE_SHA, 'contract_sha256': c.sha(contract_path),
              'wrapper_sha256': c.sha(c.WRAPPER), 'audit_path': str(audit_path), 'audit_sha256': c.sha(audit_path),
              'numerical_input_files_hashed_before_timing': verified_inputs,
              'runner_sha256': c.sha(Path(__file__)), 'common_sha256': c.sha(c.BASE / 'common.py'),
              'adaptation_record_sha256': c.sha(c.BASE / 'adaptation-record.json'),
              'frozen_sources_sha256': adaptation['frozen_sources_sha256'],
              'ce_candidate_sha256': c.sha(c.BF / 'candidates/0008/candidate.py'),
              'gradnorm_candidate_sha256': c.sha(c.GRAD / 'candidate.py'),
              'validator_adaptation': validator_adaptation, 'model': contract['model'],
              'seeds': seeds, 'orders': c.ORDERS, 'pairs': [], 'completed_cases': [],
              'promotion_rule': {'gain_vs_control_strictly_greater_than': 0.01,
                                 'all_three_seeds_faster_vs_control': True}, 'started_utc': c.utc(),
              'scope': 'Fresh same-round native/control/optimized heldout model. Frozen SwiGLU004, '
                       'CE0008 and gradnorm001; native RMSNorm/default compiled activation baseline. '
                       'Pinned original native GPU-synchronized time.time interval timer: 120 steps, '
                       'first20 excluded and ten log-interval10 intervals retained. No audit/profile/'
                       'checkpoint/eval during timing. No production, convergence or multiGPU claim.'}
    try:
        for seed, order in zip(seeds, c.ORDERS):
            arms = {}
            for arm in order:
                folder = folders[(seed, arm)]
                expected = c.command_for(folder, arm, contract_path, contract, 120, seed)
                validator.EXPECTED_COMMANDS[str(folder)] = expected
                record = c.run_case(folder, arm, args.model, contract_path, contract, 120, seed)
                result = validator.validate_case(folder, record['mode'], seed, contract,
                                                 c.sha(contract_path), c.sha(c.BF / 'candidates/0008/candidate.py'))
                result.update(stage30_arm=arm, hooks=c.hook_gate(folder, arm, contract, 120, manifest))
                c.require(tuple(row['iteration'] for row in result['intervals'] if row['iteration'] != 1)
                          == tuple(range(10, 121, 10))
                          and sum(row['retained'] for row in result['intervals']) == 10,
                          'complete original first20 exclusion/final100 interval coverage differs')
                arms[arm] = result
                report['completed_cases'].append({'arm': arm, 'seed': seed, 'case': folder.name,
                                                  'mean_iteration_ms': result['mean_iteration_ms']})
                print(c.json.dumps(report['completed_cases'][-1]), flush=True)
            c.require(arms['native']['model'] == arms['control']['model'] == arms['optimized']['model'],
                      'actual heldout models differ between arms')
            for first, second in zip(order, order[1:]):
                c.require(datetime.fromisoformat(arms[first]['finished_utc'])
                          <= datetime.fromisoformat(arms[second]['started_utc']),
                          'performance processes overlap or fixed rotation order differs')
            native, control, optimized = (arms[name]['mean_iteration_ms'] for name in
                                         ('native', 'control', 'optimized'))
            report['pairs'].append({'seed': seed, 'order': order, 'arms': arms,
                                    'vs_native': native / optimized, 'vs_control': control / optimized,
                                    'control_vs_native': native / control})
        for metric in ('vs_native', 'vs_control', 'control_vs_native'):
            ratio = math.exp(statistics.mean(math.log(pair[metric]) for pair in report['pairs']))
            report[metric] = {'ratio': ratio, 'gain_percent': 100 * (ratio - 1),
                              'all_seeds_faster': all(pair[metric] > 1 for pair in report['pairs'])}
        c.require(len(report['pairs']) == 3 and len(report['completed_cases']) == 9,
                  'complete nine-process heldout measurement required')
        report['promoted'] = report['vs_control']['ratio'] > 1.01 and report['vs_control']['all_seeds_faster']
        report.update(status='validated', decision='promoted_tested_model_only' if report['promoted'] else 'not_promoted')
    except Exception as error:
        report.update(status='rejected', promoted=False, decision='not_promoted',
                      error_type=type(error).__name__, error=str(error))
    report['finished_utc'] = c.utc()
    c.write(output, report)
    print(c.json.dumps({key: report.get(key) for key in ('status', 'decision', 'vs_native', 'vs_control', 'error')}
                       | {'output': str(output)}), flush=True)
    return 0 if report['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
