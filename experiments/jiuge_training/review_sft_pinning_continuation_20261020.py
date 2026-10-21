"""Independent closed-key review with unchanged timing and loss checks."""
import hashlib
from pathlib import Path

BASE = Path(__file__).resolve().parent
SOURCE_SHA256 = '8fbde7dd87a65728b1febec46b7f41b7a51d10f7b92948d0b90e2f8e7c1f31f1'


def derive(payload):
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError('Original independent review source changed')
    text = payload.decode('utf-8')

    def once(before, after):
        nonlocal text
        if text.count(before) != 1:
            raise ValueError('Exact independent-review derivation changed: ' + before[:90])
        text = text.replace(before, after)

    once("DRIVER = BASE/'results/sft-pinning-driver-20261020-01'",
         "DRIVER = BASE/'results/sft-pinning-formal-driver-20261020-03'")
    once("DEST = BASE/'results/sft-pinning-verified-20261020'",
         "DEST = BASE/'results/sft-pinning-verified-20261020-continuation'")
    once("    root = DEST/'sft-pinning-validation-20261020-01'",
         "    root = DEST/'sft-pinning-formal-continuation-20261020-01'\n"
         "    correctness = DEST/'sft-pinning-validation-20261020-01'")
    once("expected = read(BASE/'results/sft-pinning-source-manifest-20261020.json')['files']",
         "expected = read(BASE/'results/sft-pinning-formal-source-manifest-v3-20261020.json')['files']")
    once("    primary = read(root/'numeric/primary-archive-verification.json')",
         "    parent = read(correctness/'controller.json')\n"
         "    parent_sha = hashlib.sha256((correctness/'controller.json').read_bytes()).hexdigest()\n"
         "    allowed_errors = {'Local key gate not closed with three-hour formal reserve',\n"
         "                      'Insufficient authorized free time for next stage'}\n"
         "    if (parent['status']!='stopped' or parent.get('error_type')!='TimeoutError'\n"
         "        or parent.get('error') not in allowed_errors or parent['formal_performance_started']\n"
         "        or not parent.get('full_correctness_resume_and_primary_archive_passed')\n"
         "        or len(parent['completed_cases'])!=6 or any(x['exit_code'] for x in parent['commands'])\n"
         "        or parent_sha!=controller['correctness_parent_controller_sha256']\n"
         "        or len(parent['source_sha256'])!=29\n"
         "        or any(expected.get(k)!=v for k,v in parent['source_sha256'].items())):\n"
         "        raise ValueError('Resource handoff did not preserve a fully validated fresh SFT parent')\n"
         "    prior_gpu = read(correctness/'allocation.json')['ordered_gpu_uuids']\n"
         "    formal_gpu = read(root/'allocation.json')['ordered_gpu_uuids']\n"
         "    if set(prior_gpu)!=set(formal_gpu) or not controller['exact_original_physical_GPU_pair_required']:\n"
         "        raise ValueError('Exact original physical GPU pair was not preserved')\n"
         "    primary = read(correctness/'numeric/primary-archive-verification.json')")
    once("semantic = read(root/'semantics/semantics-verification.json')",
         "semantic = read(correctness/'semantics/semantics-verification.json')")
    once("result = analyze(DEST,'sft20261020'+name)",
         "result = analyze(DEST,'sft20261020continuation'+name)")
    once("results/sft-pinning-independent-review-20261020.json",
         "results/sft-pinning-independent-review-20261020-continuation.json")
    once("'old_CPT_84_4_percent_reused_as_SFT_result':False}",
         "'old_CPT_84_4_percent_reused_as_SFT_result':False,\n"
         "              'exact_original_physical_GPU_pair_preserved':True,\n"
         "              'complete_SFT_parent_reused_without_CPT_numeric_references':True,\n"
         "              'resource_handoff_only_no_mathematical_or_timing_standard_change':True}")
    return text


def main():
    source = derive((BASE/'review_sft_pinning_results.py').read_bytes())
    exec(compile(source, str(BASE/'review_sft_pinning_continuation_20261020.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
