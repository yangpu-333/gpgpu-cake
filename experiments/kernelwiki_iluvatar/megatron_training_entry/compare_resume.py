"""Compare the real original Megatron continuation iterations four through six.

Reuse the frozen three-step tensor comparison and tolerance gates, with an
explicit forward-index-to-training-iteration mapping. Checkpoint files are
hashed, not unpickled. No training, candidate, or GPU code is executed here.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import compare_cases as base
from run_resume import checkpoint_manifest, digest, verify_resume_execution


ROOT = Path(__file__).resolve().parent


def iteration_label(name):
    def replace(match):
        return match.group(1) + str(int(match.group(2)) + 3)
    return re.sub(r'^(?P<prefix>(?:(?:native|optimized) )?step )([123])\b', replace, name)


class ResumeComparison(base.Comparison):
    def __init__(self, native, optimized, checkpoint_case, torch_module):
        super().__init__(native, optimized, torch_module)
        self.checkpoint_case = Path(checkpoint_case)
        self.resume_proofs = {}
        self.source_checkpoint_files = {}

    def require(self, condition, name, **details):
        return super().require(condition, iteration_label(name), **details)

    def compare_tree(self, native, optimized, name, group, exact=False):
        return super().compare_tree(native, optimized, iteration_label(name), group, exact)

    def compare_tensor(self, native, optimized, name, group, exact=False):
        return super().compare_tensor(native, optimized, iteration_label(name), group, exact)

    def read_snapshot(self, arm, relative):
        match = re.fullmatch(r'(wgrads|params)/iter_000000([123])/mp_rank_00\.pth', str(relative))
        if match:
            relative = '%s/iter_%07d/mp_rank_00.pth' % (match.group(1), int(match.group(2)) + 3)
        return super().read_snapshot(arm, relative)

    def native_parameter_dump(self, native, optimized, initial, metadata, step, label):
        return super().native_parameter_dump(native, optimized, initial, metadata, step + 3, label)

    def derived_loss(self, native, optimized, step):
        return super().derived_loss(native, optimized, step + 3)

    def metadata(self):
        records = super().metadata()
        if records is None:
            return None
        directory = self.checkpoint_case / 'snapshots'
        source_files = checkpoint_manifest(directory, 3)
        self.source_checkpoint_files = source_files
        for relative in source_files:
            self.inventory(directory / relative, 'common-native-checkpoint')
        source_command_path = self.checkpoint_case / 'command.json'
        source_entry_path = self.checkpoint_case / 'entry.json'
        source_command = json.loads(source_command_path.read_text(encoding='utf-8'))
        source_entry = json.loads(source_entry_path.read_text(encoding='utf-8'))
        self.inventory(source_command_path, 'common-native-checkpoint')
        self.inventory(source_entry_path, 'common-native-checkpoint')
        self.require(source_command.get('mode') == 'native' and source_command.get('audit') is True
                     and source_command.get('steps') == 3 and source_command.get('exit_code') == 0
                     and source_entry.get('status') == 'completed'
                     and source_entry.get('iluvatar_kernels') is False
                     and source_entry.get('kernel_adapters') == [],
                     'common input checkpoint comes from completed native three-step audit')
        self.require(source_command.get('contract_sha256') == digest(ROOT / 'contract.json'),
                     'common native checkpoint uses fixed contract')
        source_records = []
        for arm, folder in (('native', self.native), ('optimized', self.optimized)):
            command, entry = records[arm]['command.json'], records[arm]['entry.json']
            self.require(command.get('resumed_from_iteration') == 3 and command.get('total_train_iters') == 6,
                         arm + ' original continuation is iteration three to six')
            self.require(command.get('resume_runner_sha256') == digest(ROOT / 'run_resume.py'),
                         arm + ' continuation runner SHA256 matches verifier')
            self.require(command.get('source_checkpoint_unchanged') is True and command.get('passed') is True,
                         arm + ' runner verified checkpoint immutability and real continuation')
            source_record = command.get('source_checkpoint', {})
            source_records.append(source_record)
            self.require(source_record.get('iteration') == 3 and source_record.get('files_sha256') == source_files,
                         arm + ' exact common input checkpoint byte hashes')
            self.require(source_record.get('source_command_sha256') == digest(source_command_path)
                         and source_record.get('source_entry_sha256') == digest(source_entry_path),
                         arm + ' common native checkpoint case metadata hashes')
            self.require(command.get('seed') == source_command.get('seed'), arm + ' original checkpoint seed preserved')
            argv = entry.get('training_argv', [])
            for flag in ('--exit-on-missing-checkpoint', '--use-checkpoint-opt-param-scheduler'):
                self.require(flag in argv, arm + ' required restore flag ' + flag)
            for flag in ('--finetune', '--no-load-optim', '--no-load-rng', '--override-ckpt-iteration',
                         '--override-opt-param-scheduler'):
                self.require(flag not in argv, arm + ' retains original restore semantics without ' + flag)
            self.require(argv.count('--load') == 1, arm + ' exactly one native checkpoint input')
            if argv.count('--load') == 1:
                load_index = argv.index('--load') + 1
                load_directory = argv[load_index] if load_index < len(argv) else ''
            else:
                load_directory = ''
            self.require(load_directory == source_record.get('directory'), arm + ' argv loads recorded common checkpoint')
            for name in ('stdout.log', 'stderr.log'):
                self.inventory(folder / name, arm)
            proof = verify_resume_execution(folder, load_directory)
            self.resume_proofs[arm] = proof
            self.require(proof['passed'], arm + ' raw native log and final checkpoint prove real resumed training',
                         errors=proof['errors'], actual_progress=proof['actual_training_progress'])
            self.require(command.get('resume_validation') == proof,
                         arm + ' independently reproduced runner continuation validation')
            for relative in proof['final_checkpoint_files_sha256']:
                self.inventory(folder / 'snapshots' / relative, arm + '-final-checkpoint')
        self.require(source_records[0] == source_records[1], 'both continuations use identical checkpoint provenance')
        self.require(records['native']['entry.json'].get('native_sources_sha256') ==
                     source_entry.get('native_sources_sha256'), 'continuations preserve native checkpoint source hashes')
        self.require(records['native']['entry.json'].get('integration_sha256') ==
                     source_entry.get('integration_sha256'), 'continuations preserve checkpoint integration hashes')
        return records

    def snapshot_layout(self):
        expected = {'model0-initial.pt'} | {'model0-forward-%03d.pt' % index for index in (1, 2, 3)}
        for arm, folder in (('native', self.native), ('optimized', self.optimized)):
            directory = folder / 'snapshots'
            observed = {path.name for path in directory.glob('model*-*.pt')}
            self.require(observed == expected, arm + ' exact pre-load and resumed forward snapshot inventory',
                         expected=sorted(expected), observed=sorted(observed))
            for label in ('wgrads', 'params'):
                observed = {str(path.relative_to(directory / label)).replace('\\', '/')
                            for path in (directory / label).glob('iter_*/*') if path.is_file()}
                expected_files = {'iter_%07d/mp_rank_00.pth' % step for step in (4, 5, 6)}
                self.require(observed == expected_files, arm + ' exact native resumed ' + label + ' inventory',
                             expected=sorted(expected_files), observed=sorted(observed))

    def run(self):
        report = super().run()
        report.update(schema='megatron-formal-entry-checkpoint-resume-comparison-v1',
                      resume_comparator_sha256=digest(__file__), resume_runner_sha256=digest(ROOT / 'run_resume.py'),
                      checkpoint_case=str(self.checkpoint_case.resolve()),
                      common_input_checkpoint_files_sha256=self.source_checkpoint_files,
                      resumed_from_iteration=3, final_iteration=6,
                      forward_to_training_iteration={'1': 4, '2': 5, '3': 6},
                      resume_execution_proofs=self.resume_proofs,
                      initial_snapshot_scope='model0-initial.pt is captured before native checkpoint loading. '
                                             'It verifies equal fresh construction, not restored parameter state. '
                                             'Original successful-load logs, identical checkpoint byte hashes, '
                                             'and actual iterations four through six establish continuation scope.',
                      performance_scope='Correctness-only checkpoint continuation with tensor/disk snapshots; '
                                        'no throughput claim is derived from these runs.')
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native_case', type=Path)
    parser.add_argument('optimized_case', type=Path)
    parser.add_argument('--checkpoint-case', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite continuation comparison evidence')
    if args.native_case.resolve() == args.optimized_case.resolve():
        parser.error('two separate continuation cases are required')
    import torch
    comparison = ResumeComparison(args.native_case, args.optimized_case, args.checkpoint_case, torch)
    try:
        report = comparison.run()
    except Exception as error:
        comparison.require(False, 'continuation comparison completed without processing errors',
                           error_type=type(error).__name__, error=str(error))
        report = {'schema': 'megatron-formal-entry-checkpoint-resume-comparison-v1', 'passed': False,
                  'resume_comparator_sha256': digest(__file__), 'checks': comparison.checks,
                  'tensor_comparisons': comparison.tensors, 'input_files': comparison.files,
                  'error_type': type(error).__name__, 'error': str(error)}
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'passed': report['passed'], 'output': str(args.output.resolve()),
                      'check_summary': report.get('check_summary')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
