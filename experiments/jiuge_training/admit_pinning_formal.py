"""Tie ACT full-value verification to the byte-verified local key-result bundle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def admit(primary, semantics, key_bundle, local_key_receipt):
    numeric = json.loads(primary.read_text())
    semantic = json.loads(semantics.read_text())
    local = json.loads(local_key_receipt.read_text())
    if not numeric['passed'] or not numeric['primary_complete_raw_values_verified'] or not numeric['checkpoint_resume_tested']:
        raise ValueError('Full ACT original values and resume verification required')
    if not semantic['passed'] or not semantic['full_model_native_checkpoint_resume_verified'] or not semantic['tp_actual_inputs_equal']:
        raise ValueError('Actual inputs, loss, RNG, scheduler and full native resume required')
    if local.get('schema') != 'jiuge-pinning-local-key-verification-v1' or not local['passed']:
        raise ValueError('Actual local key verification receipt required')
    if local['bundle_sha256'] != sha(key_bundle) or local['full_raw_local_copy_verified']:
        raise ValueError('Key bundle differs or falsely claims a complete local original archive')
    with zipfile.ZipFile(key_bundle) as archive:
        catalog = json.loads(archive.read('key-results-manifest.json'))
        for path in (primary, semantics):
            relative = path.resolve().relative_to(Path(catalog['primary_root']).resolve()).as_posix()
            if catalog['files'][relative]['sha256'] != sha(path) or hashlib.sha256(archive.read(relative)).hexdigest() != sha(path):
                raise ValueError('Local key verification does not cover current complete validation receipts')
    return {'schema': 'jiuge7b-pinning-formal-admission-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'passed': True, 'full_numeric_comparison_verified': True,
            'primary_complete_raw_values_verified': True, 'full_model_native_checkpoint_resume_verified': True,
            'inputs_masks_rng_scheduler_verified': True, 'local_key_results_verified': True,
            'local_full_raw_copy_required': False, 'local_full_raw_copy_verified': False,
            'candidate_sha256': '8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88',
            'verified_receipts': [{'file': str(path.resolve()), 'sha256': sha(path)}
                                  for path in (primary, semantics, local_key_receipt)],
            'formal_performance_passed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--semantics', type=Path, required=True)
    parser.add_argument('--key-bundle', type=Path, required=True)
    parser.add_argument('--local-key-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = admit(args.primary, args.semantics, args.key_bundle, args.local_key_receipt)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)
