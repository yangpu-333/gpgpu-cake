"""Confirm immutable002 remains rejected, with nonpromoted raw-log diagnostics.

This is not the complete archive verifier. It accepts only the exact preserved
metadata-loop bug, confirms the normal verifier rejects it, and separately
recomputes the already nonpromoted timing result. No evidence is rewritten.
"""
import argparse
import json
from pathlib import Path
import sys
import verify_receipts as v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('002',), default='002')
    parser.add_argument('--base', type=Path, default=Path(__file__).resolve().parent / 'evidence')
    parser.add_argument('--audit-suffix', default='complete')
    parser.add_argument('--run-suffix', default='complete')
    args = parser.parse_args()
    args.probe = None
    sys.dont_write_bytecode = True
    try:
        archive = v.Archive(args)
        archive.foundations()
        archive.backup()
        path, comparisons, routes = archive.audit()
        rejected = False
        try:
            archive.performance(path, routes)
        except ValueError as error:
            rejected = str(error) == 'performance numerical audit-summary path/SHA binding differs'
        v.require(rejected, 'complete verifier must still reject the exact002 audit metadata binding')
        performance = archive.performance(path, routes, diagnose_metadata_failure=True)
        v.require(performance['promoted'] is False and performance['audit_metadata_binding_passed'] is False
                  and performance['diagnostic_only'] is True and performance['vs_control']['all_seeds_faster'] is False,
                  'original002 must remain rejected and nonpromoted')
        result = {'status': 'expected_rejection_confirmed', 'version': '002',
                  'complete_archive_status': 'rejected', 'reason': 'numerical audit-summary path/SHA metadata binding',
                  'promoted': False, 'comparisons': comparisons, 'diagnostic_performance': performance,
                  'snapshot_files_available': archive.snapshot_present,
                  'snapshot_files_missing': archive.snapshot_missing,
                  'tensor_arithmetic_replayed': False,
                  'verifier_sha256': v.sha(__file__), 'gpu_rerun': False, 'evidence_rewritten': False,
                  'scope': 'Original002 preserved. Full numerical receipts and source/header/timing data rechecked '
                           'only to diagnose known metadata rejection and NOTpromoted result. '
                           'This does not convert002 into a completely verified optimization round.'}
    except Exception as error:
        result = {'status': 'rejected', 'error_type': type(error).__name__, 'error': str(error), 'gpu_rerun': False}
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] == 'expected_rejection_confirmed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
