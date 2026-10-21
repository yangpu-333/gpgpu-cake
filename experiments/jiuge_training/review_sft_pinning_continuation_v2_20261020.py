"""Recover the postprocessing-only first-step logging assumption.

Scientific sources, raw logs, timing and tolerances remain unchanged.
Preserve the failed review and verify its existing extraction byte for byte.
Check all twelve regular loss intervals plus Megatron's extra first step.
"""
import hashlib
import math
from pathlib import Path
import re

from review_sft_pinning_continuation_20261020 import derive as previous_derive

BASE = Path(__file__).resolve().parent
EXPECTED_ITERATIONS = [1, *range(10, 121, 10)]


def logged_loss_records(text):
    records = []
    for line in text.splitlines():
        match = re.search(r'\biteration\s+(\d+)\s*/\s*120\s*\|.*?\blm loss:\s*([^\s|]+)', line)
        if match:
            value = float(match.group(2))
            if not math.isfinite(value):
                raise ValueError('Non-finite formal logged loss')
            for name in ('skipped', 'nan'):
                count = re.search(r'number of '+name+r' iterations:\s*(\d+)', line)
                if count is None or int(count.group(1)) != 0:
                    raise ValueError('Missing/nonzero formal '+name+' iteration count')
            records.append((int(match.group(1)), value))
    if ([row[0] for row in records] != EXPECTED_ITERATIONS or
            len(re.findall(r'\blm loss:', text)) != len(EXPECTED_ITERATIONS)):
        raise ValueError('Require first step and all twelve regular formal logged loss intervals')
    return records


def derive(payload):
    source = previous_derive(payload)

    def once(before, after):
        nonlocal source
        if source.count(before) != 1:
            raise ValueError('Exact postprocessing recovery boundary changed: '+before[:100])
        source = source.replace(before, after)

    once("    if status['status'] != 'completed':\n",
         "    if (status['status'] != 'independent-review-failed'\n"
         "        or status.get('remote_status') != 'completed'\n"
         "        or status.get('independent_review_completed')\n"
         "        or status.get('exit_code') != 1):\n")
    once("DRIVER/'independent-final-key-verification.json'",
         "DRIVER/'independent-final-key-verification-v2.json'")
    once("    DEST.mkdir(exist_ok=False)\n"
         "    with zipfile.ZipFile(DRIVER/'sft-complete-key-results.zip') as archive:\n"
         "        archive.extractall(DEST)  # verify() checked every member/path/size/SHA above.\n",
         "    prior = read(DRIVER/'final-key-verification.json')\n"
         "    for key in ('passed','bundle_sha256','verified_key_files','verified_key_bytes'):\n"
         "        if prior[key] != checked[key]:\n"
         "            raise ValueError('Closed final key receipt differs: '+key)\n"
         "    if not DEST.is_dir():\n"
         "        raise ValueError('Preserve and reuse the original failed-review extraction')\n"
         "    with zipfile.ZipFile(DRIVER/'sft-complete-key-results.zip') as archive:\n"
         "        if {p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()} != set(archive.namelist()):\n"
         "            raise ValueError('Existing extraction membership differs')\n"
         "        for member in archive.namelist():\n"
         "            path = DEST/member\n"
         "            if path.is_symlink() or not path.resolve().is_relative_to(DEST.resolve()):\n"
         "                raise ValueError('Existing extracted member is not a regular in-root file')\n"
         "            if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(archive.read(member)).digest():\n"
         "                raise ValueError('Existing extracted key bytes differ: '+member)\n")
    once("                losses = [[float(value) for value in re.findall(r'lm loss:\\s*([\\d.E+\\-]+)',text)] for text in logs]\n"
         "                if any(len(values)!=12 or not all(math.isfinite(x) for x in values) for values in losses):\n"
         "                    raise ValueError('All twelve formal logged loss intervals must be finite')\n",
         "                records = [logged_loss_records(text) for text in logs]\n"
         "                losses = [[value for iteration,value in rows] for rows in records]\n")
    once("'intervals':12,", "'intervals':13,'regular_intervals':12,\n"
         "                                    'logged_iterations':[row[0] for row in records[0]],\n"
         "                                    'extra_first_step_checked':True,")
    once("results/sft-pinning-independent-review-20261020-continuation.json",
         "results/sft-pinning-independent-review-20261020-continuation-v2.json")
    once("'resource_handoff_only_no_mathematical_or_timing_standard_change':True}",
         "'resource_handoff_only_no_mathematical_or_timing_standard_change':True,\n"
         "              'postprocessing_recovery':'Megatron logs iteration1 in addition to 10..120; all13 checked',\n"
         "              'failed_original_review_preserved':True,\n"
         "              'existing_extracted_keys_reverified_without_overwrite':True}")
    return source


def main():
    source = derive((BASE/'review_sft_pinning_results.py').read_bytes())
    exec(compile(source, str(BASE/'review_sft_pinning_continuation_v2_20261020.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__, 'logged_loss_records': logged_loss_records})


if __name__ == '__main__':
    main()
