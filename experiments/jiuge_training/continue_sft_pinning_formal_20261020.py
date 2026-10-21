"""Exact formal-only continuation after the original free-time gate closes.

The original 29 scientific sources and all SFT originals stay immutable.
This wrapper derives the already-frozen controller, preserving its complete
formal protocol and requiring the exact original physical GPU pair.
"""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
ORIGINAL_SHA256 = 'af243eb7140014b3bf37cb1379e590ad37dc1889f4e0e23e09c5a94694844968'


def derive(payload):
    if hashlib.sha256(payload).hexdigest() != ORIGINAL_SHA256:
        raise ValueError('Original frozen SFT controller changed')
    text = payload.decode('utf-8')

    def once(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError('Exact continuation derivation changed: ' + before[:100])
        text = text.replace(before, after)

    once("ROOT = PREP/'sft-pinning-validation-20261020-01'",
         "ROOT = PREP/'sft-pinning-formal-continuation-20261020-01'\n"
         "CORRECTNESS = PREP/'sft-pinning-validation-20261020-01'")
    once("NUMERIC = ROOT/'numeric'", "NUMERIC = CORRECTNESS/'numeric'")
    once("SEMANTICS = ROOT/'semantics'", "SEMANTICS = CORRECTNESS/'semantics'")
    once("CODE = ('continue_sft_pinning_validation.py',",
         "CODE = ('continue_sft_pinning_formal_20261020.py',"
         "'bootstrap_sft_pinning_formal_20261020.sh','continue_sft_pinning_validation.py',")
    once("approval = read(PREP/'sft-pinning-launch-approved-20261020.json')",
         "approval = read(PREP/'sft-pinning-formal-approved-20261020.json')")
    once("        expected_uuids = uuids\n",
         "        prior_uuids = read(CORRECTNESS/'allocation.json')['ordered_gpu_uuids']\n"
         "        if set(uuids) != set(prior_uuids):\n"
         "            raise ValueError('Exact original SFT physical GPU pair required; no timing on replacement GPUs')\n"
         "        expected_uuids = uuids\n")

    start = "        command('prepare-numeric',[PREP/'pinning_candidate.py','--output',NUMERIC])\n"
    end = "        from admit_pinning_formal import admit\n"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError('Original correctness/formal boundary changed')
    left, right = text.index(start), text.index(end)
    if left >= right:
        raise ValueError('Invalid frozen phase ordering')
    proof_gate = '''        from pinning_key_results import export
        parent = read(CORRECTNESS/'controller.json')
        expected_cases = [
            'sft-pinning-audit-native-20261020-01',
            'sft-pinning-audit-control-20261020-01',
            'sft-pinning-audit-optimized-20261020-01',
            'sft-pinning-native-checkpoint-20261020-01',
            'sft-pinning-native-resume-20261020-01',
            'sft-pinning-optimized-resume-20261020-01']
        allowed_errors = {
            'Local key gate not closed with three-hour formal reserve',
            'Insufficient authorized free time for next stage'}
        if (parent['status'] != 'stopped' or parent.get('error_type') != 'TimeoutError'
            or parent.get('error') not in allowed_errors
            or parent.get('formal_performance_started')
            or not parent.get('full_correctness_resume_and_primary_archive_passed')
            or parent['completed_cases'] != expected_cases
            or any(row['exit_code'] for row in parent['commands'])):
            raise ValueError('Only a fully validated SFT parent stopped by its time reserve can be continued')
        for name, digest in parent['source_sha256'].items():
            if hashes.get(name) != digest or sha(CORRECTNESS/'source'/name) != digest:
                raise ValueError('Parent/current scientific source identity differs: '+name)
        if len(parent['source_sha256']) != 29:
            raise ValueError('Complete original29 source identity required')
        for name in expected_cases:
            folder = PREP/'runs'/name
            if read(folder/'case.json')['exit_code']:
                raise ValueError('Parent SFT case did not close successfully')
            cases.append(folder)
        primary = NUMERIC/'primary-archive-verification.json'
        semantic_receipt = SEMANTICS/'semantics-verification.json'
        semantic = read(semantic_receipt)
        if (not semantic['passed'] or semantic.get('objective') != 'answer_only_SFT'
            or not semantic.get('sft_mask_gate')):
            raise ValueError('Actual fresh SFT semantics required; CPT proof is ineligible')
        checkpoint_inventory = read(CORRECTNESS/'checkpoint-sha256.json')
        if not checkpoint_inventory['files']:
            raise ValueError('Native SFT checkpoint SHA inventory required')
        bundle = PREP/'sft-pinning-correctness-key-results-20261020.zip'
        if parent['key_results_bundle'] != str(bundle):
            raise ValueError('Parent correctness key bundle identity differs')
        local_receipt = CORRECTNESS/'local-key-verification.json'
        update('waiting-for-parent-local-key-verification')
        while not local_receipt.exists():
            frozen(10800)
            time.sleep(5)
        state.update(correctness_parent=str(CORRECTNESS),
                     correctness_parent_controller_sha256=sha(CORRECTNESS/'controller.json'),
                     full_correctness_resume_and_primary_archive_passed=True,
                     exact_original_physical_GPU_pair_required=True)
'''
    text = text[:left] + proof_gate + text[right:]
    once("label = 'sft20261020'+round_name", "label = 'sft20261020continuation'+round_name")
    once("final = PREP/'sft-pinning-complete-key-results-20261020.zip'",
         "final = PREP/'sft-pinning-complete-key-results-20261020-continuation.zip'")
    once("export(PREP,[ROOT,*cases],final,64*1024**2)",
         "export(PREP,[ROOT,CORRECTNESS,*cases],final,64*1024**2)")
    # No replacement touches seeds, arm order, steps, warmup, timer, admission,
    # analysis or the mandatory CV confirmation policy below the phase boundary.
    return text


def main():
    payload = (PREP/'continue_sft_pinning_validation.py').read_bytes()
    source = derive(payload)
    exec(compile(source, str(PREP/'continue_sft_pinning_formal_20261020.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
