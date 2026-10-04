"""Nine original synchronized training processes after complete Attention audit."""
import argparse
from datetime import datetime
import math
from pathlib import Path
import statistics
import common as c


def numerical_gate(version, suffix, contract_path, contract, manifest, provenance):
    path = c.BASE / ('audit-summary-main-' + version + '-' + suffix + '.json')
    audit = c.read(path)
    c.require(audit['schema'] == 'native-full-attention-stage31-three-way-audit-v1'
              and audit['status'] == 'completed' and audit['passed'] is True
              and audit['version'] == version and audit['model_name'] == 'main' and audit['suffix'] == suffix
              and audit['audit_seed'] == c.AUDIT_SEED
              and audit['candidate_sha256'] == provenance['candidate_sha256']
              and audit['contract_sha256'] == c.sha(contract_path)
              and audit['wrapper_sha256'] == c.sha(c.WRAPPER)
              and audit['runner_sha256'] == c.sha(c.BASE / 'run_audit.py')
              and audit['controller_sources_sha256'] == c.sources() and audit['provenance'] == provenance
              and audit['frozen_comparator_sha256'] == c.sha(c.BF / 'compare_training.py'),
              'unaltered passing same-version three-way complete numerical audit required')
    c.require(set(audit['comparisons']) == {'controlled', 'native_optimized', 'native_control'}
              and len(audit['completed_cases']) == 3
              and {item['arm'] for item in audit['completed_cases']} == {'native', 'control', 'optimized'},
              'complete three-arm numerical audit required')
    inventories = {}
    for label, item in audit['comparisons'].items():
        comparison = Path(item['path'])
        proof = c.read(comparison)
        c.require(comparison.resolve().parent == c.BASE.resolve() and c.sha(comparison) == item['sha256']
                  and item['passed'] is True and proof['passed'] is True
                  and proof['stage31_comparison'] == label and proof['model_name'] == 'main'
                  and proof['version'] == version and proof['steps'] == 3
                  and proof['atol'] == 3e-4 and proof['rtol'] == 1e-3
                  and proof['check_summary']['failed'] == 0
                  and all(check['passed'] is True for check in proof['checks'])
                  and all(tensor['passed'] is True for tensor in proof['tensor_comparisons'])
                  and proof['comparator_sha256'] == audit['frozen_comparator_sha256']
                  and proof['stage31_runner_sha256'] == audit['runner_sha256']
                  and proof['stage31_controller_sources_sha256'] == c.sources()
                  and proof['stage31_probe_sha256'] == provenance['probe_sha256']
                  and proof['attention_candidate_sha256'] == provenance['candidate_sha256'],
                  'complete frozen numerical comparison differs: ' + label)
        computed = {'total': len(proof['checks']),
                    'passed': sum(check['passed'] is True for check in proof['checks']),
                    'failed': sum(check['passed'] is not True for check in proof['checks'])}
        c.require(proof['check_summary'] == computed and computed['total'] > 0
                  and bool(proof['tensor_comparisons']) and item['check_summary'] == computed
                  and item['tensor_comparisons'] == len(proof['tensor_comparisons'])
                  and item['all_tensors_exact_equal'] == all(row['exact_equal'] for row in proof['tensor_comparisons'])
                  and item['all_tensor_bytes_equal'] == all(row['native_sha256'] == row['optimized_sha256']
                                                           for row in proof['tensor_comparisons']),
                  'numerical summary must reflect every raw check and tensor record')
        c.require(bool(proof['input_files']), 'frozen numerical input inventory missing')
        for row in proof['input_files']:
            raw = Path(row['path']).resolve()
            c.require(any(raw.is_relative_to(root.resolve()) for root in
                          (c.BASE, c.BF, c.FORMAL, c.GRAD, c.SEALED)),
                      'numerical input inventory escaped fixed experiment/source roots')
            previous = inventories.setdefault(str(raw), row)
            c.require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                      'same numerical raw input has conflicting inventory metadata')
    for raw_name, row in inventories.items():
        raw = Path(raw_name)
        c.require(raw.is_file() and raw.stat().st_size == row['bytes'] and c.sha(raw) == row['sha256'],
                  'raw tensor/metadata bytes changed after full numerical audit: ' + raw_name)
    routes = {}
    for item in audit['completed_cases']:
        folder = Path(item['path'])
        c.require(folder.resolve().parent == c.BASE.resolve(), 'unexpected audited case path')
        current = c.hook_gate(folder, item['arm'], version, contract, 3, manifest, provenance['backend'])
        c.require(item['hooks'] == current, 'raw numerical runtime receipts differ from sealed audit')
        observation = current['attention']
        routes[item['arm']] = {
            'case': folder.name,
            'attention_receipt_sha256': current['raw_sha256']['attention.json'],
            'native_routes_observed': observation['native_routes_observed'],
            'native_fused_forward_calls': observation['native_fused_forward_calls'],
            'native_fused_backward_calls': observation['native_fused_backward_calls'],
            'native_torch_forward_calls': observation['native_torch_forward_calls'],
            'candidate_calls': observation['candidate_calls'],
            'fallback_calls': observation['fallback_calls']}
    return path, len(inventories), routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    args = parser.parse_args()
    c.require(all(value.replace('-', '').isalnum() for value in (args.audit_suffix, args.run_suffix)),
              'simple immutable audit/run suffixes required')
    contract_path, contract, manifest, provenance = c.prepare(args.version, args.probe)
    audit_path, verified_inputs, audit_routes = numerical_gate(
        args.version, args.audit_suffix, contract_path, contract, manifest, provenance)
    seeds = contract['performance']['independent_seeds']
    output = c.BASE / ('performance-main-' + args.version + '-' + args.run_suffix + '.json')
    folders = {(seed, arm): c.BASE / ('main-' + arm + '-perf-' + args.version + '-' + args.run_suffix + '-' + str(seed))
               for seed in seeds for arm in ('native', 'control', 'optimized')}
    c.require(not any(path.exists() for path in [output, *folders.values()]),
              'refusing to overwrite immutable Attention performance evidence')
    validator, validator_adaptation = c.load_performance_validator()
    report = {'schema': 'native-full-attention-stage31-three-arm-performance-v1', 'status': 'running',
              'promoted': False, 'version': args.version, 'model_name': 'main', 'run_suffix': args.run_suffix,
              'candidate_sha256': provenance['candidate_sha256'], 'contract_sha256': c.sha(contract_path),
              'wrapper_sha256': c.sha(c.WRAPPER), 'audit_path': str(audit_path), 'audit_sha256': c.sha(audit_path),
              'numerical_input_files_hashed_before_timing': verified_inputs,
              'untimed_audit_route_provenance': audit_routes,
              'performance_native_routes_observed_per_call': False,
              'runner_sha256': c.sha(Path(__file__)), 'controller_sources_sha256': c.sources(),
              'provenance': provenance, 'ce_candidate_sha256': c.sha(c.BF / 'candidates/0008/candidate.py'),
              'gradnorm_candidate_sha256': c.sha(c.GRAD / 'candidate.py'), 'swiglu_candidate_sha256': c.SWIGLU_SHA,
              'validator_adaptation': validator_adaptation, 'model': contract['model'],
              'seeds': seeds, 'orders': c.ORDERS, 'pairs': [], 'completed_cases': [],
              'promotion_rule': {'gain_vs_control_strictly_greater_than': 0.01,
                                 'all_three_seeds_faster_vs_control': True}, 'started_utc': c.utc(),
              'scope': 'Fresh same-round native/control/optimized four-layer BF16 GPT. '
                       'Control freezes CE0008/gradnorm001/SwiGLU004; optimized only adds complete Attention. '
                       'Default native fused softmax enabled, native RMSNorm. Original GPU-synchronized '
                       'time.time interval timer:120steps,first20excluded,ten log-interval10 intervals retained. '
                       'No tensor capture/audit/profile/checkpoint/eval during timing. Native fused/tensor '
                       'fallback route APIs are proved by the separately bound untimed audit. Performance '
                       'has equal main-dispatch coverage observation in all arms, actual native configuration '
                       'and binary provenance, with no extra native softmax forward/backward wrappers and '
                       'no per-call native fused route observation or GPU timeline claim. '
                       'No heldout, production, convergence or multiGPU claim.'}
    try:
        for seed, order in zip(seeds, c.ORDERS):
            arms = {}
            for arm in order:
                folder = folders[(seed, arm)]
                expected = c.command_for(folder, arm, args.version, contract_path, contract, 120, seed)
                validator.EXPECTED_COMMANDS[str(folder)] = expected
                record = c.run_case(folder, arm, args.version, contract_path, contract, 120, seed)
                result = validator.validate_case(folder, record['mode'], seed, contract,
                                                 c.sha(contract_path), c.sha(c.BF / 'candidates/0008/candidate.py'))
                result.update(stage31_arm=arm, hooks=c.hook_gate(folder, arm, args.version, contract,
                                                               120, manifest, provenance['backend']))
                c.require(tuple(row['iteration'] for row in result['intervals'] if row['iteration'] != 1)
                          == tuple(range(10, 121, 10))
                          and sum(row['retained'] for row in result['intervals']) == 10,
                          'complete first20 exclusion/final100 original timer interval coverage differs')
                arms[arm] = result
                report['completed_cases'].append({'arm': arm, 'seed': seed, 'case': folder.name,
                                                  'mean_iteration_ms': result['mean_iteration_ms']})
                print(c.json.dumps(report['completed_cases'][-1]), flush=True)
            c.require(arms['native']['model'] == arms['control']['model'] == arms['optimized']['model'],
                      'actual models differ between arms')
            for first, second in zip(order, order[1:]):
                c.require(datetime.fromisoformat(arms[first]['finished_utc'])
                          <= datetime.fromisoformat(arms[second]['started_utc']),
                          'performance processes overlap or fixed rotation differs')
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
                  'complete nine-process Attention measurement required')
        report['promoted'] = report['vs_control']['ratio'] > 1.01 and report['vs_control']['all_seeds_faster']
        report.update(status='validated', decision='promoted_tested_model_only' if report['promoted'] else 'not_promoted')
    except Exception as error:
        report.update(status='rejected', promoted=False, decision='not_promoted',
                      error_type=type(error).__name__, error=str(error))
    report['finished_utc'] = c.utc()
    c.write(output, report)
    print(c.json.dumps({key: report.get(key) for key in
                       ('status', 'decision', 'vs_native', 'vs_control', 'error')}
                       | {'output': str(output)}), flush=True)
    return 0 if report['status'] == 'validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
