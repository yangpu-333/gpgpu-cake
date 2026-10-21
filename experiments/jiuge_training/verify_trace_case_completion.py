"""Check actual native updates, inputs and complete capture receipts on ACT."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--sft', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    root = args.case_root
    def read(path):
        return json.loads(path.read_text())
    case = read(root/'case.json')
    if case['exit_code'] != 0:
        raise ValueError('Successful case exit required')
    expected = [{'step': step, 'skipped': 0} for step in range(1,args.steps+1)]
    for rank in range(2):
        entry = read(root/f'rank-{rank}/entry.json')
        if entry['status'] != 'completed' or entry['native_files_checked'] != 600:
            raise ValueError('Native source/entry verification failed')
        if entry['completed_native_steps'] != expected:
            raise ValueError('Actual non-skipped native step sequence differs')
        weights=entry['weight_loading']
        if (weights['source_tensor_count'] != 339 or weights['source_parameter_elements'] != 7668831744
            or not weights['all_source_tensors_and_native_parameters_covered']
            or not weights['all_loaded_shard_bytes_verified']):
            raise ValueError('Actual official full-model weights were not byte-verified')
        leaves = entry['actual_optimizer_state']['optimizers']
        if sum(item['adam_state_entries'] for item in leaves) != 199:
            raise ValueError('Full-parameter AdamW state coverage differs')
        if any(item['min_step'] != args.steps or item['max_step'] != args.steps
               for item in leaves if item['adam_state_entries']):
            raise ValueError('Actual AdamW state counter differs')
        if (sum(item['elements_with_master_param'] for item in leaves) != 3834518016
            or sum(item['elements_with_both_adam_moments'] for item in leaves) != 3834518016):
            raise ValueError('Actual master/moment element coverage differs from complete TP shard')
        if not args.profile:
            inputs = read(root/f'rank-{rank}/actual-inputs.json')
            if len(inputs) != 3:
                raise ValueError('Three actual admission batches required')
            for batch in inputs:
                path = root/f'rank-{rank}'/batch['raw_file']
                if hashlib.sha256(path.read_bytes()).hexdigest() != batch['raw_sha256']:
                    raise ValueError('Admission raw input SHA differs')
                if not batch['labels_shift_matches_tokens_inside_sequence']:
                    raise ValueError('Next-token labels do not shift correctly')
                if not args.sft and not batch['loss_mask_all_ones']:
                    raise ValueError('CPT must retain full-text supervision')
    if args.sft:
        rows = read(root/'sft-supervision-rank-0.json')['rows']
        if (len(rows) != args.steps
            or any(not 0 < item['supervised_answer_and_EOD_targets'] <= 2048 for item in rows)
            or not any(item['supervised_answer_and_EOD_targets'] < 2048 for item in rows)):
            raise ValueError('SFT must actually consume answer-only batches for every step')
        if not args.profile and not read(root/'sft-admission-verification.json')['passed']:
            raise ValueError('Independent original-document SFT input admission required')
    captures = []
    if args.profile:
        for window, steps in ((1,list(range(21,26))),(2,list(range(31,36)))):
            for rank in range(2):
                receipt = read(root/f'window-{window:02d}/rank-{rank}-capture.json')
                if (receipt['stable_step_ids'] != steps or not receipt['whole_window_boundary_drained']
                    or receipt['gpu_collective_kernel_events'] <= 0
                    or receipt['recorded_kernel_launches'] != receipt['gpu_kernel_events']
                    or receipt['launch_ids_without_gpu_kernel']):
                    raise ValueError('Whole-step GPU/NCCL capture receipt failed')
                captures.append(receipt)
    losses = [float(x) for x in re.findall(r'lm loss:\s*([\d.E+\-]+)',(root/'stdout.log').read_text())]
    if len(losses) != args.steps or not all(math.isfinite(value) for value in losses):
        raise ValueError('Finite native loss required for every actual update')
    result = {'schema': 'jiuge7b-trace-case-completion-v1',
              'utc':datetime.now(timezone.utc).isoformat(), 'case':case['case'], 'passed':True,
              'objective':'answer_only_SFT' if args.sft else 'whole_text_CPT',
              'actual_non_skipped_updates_each_rank':args.steps, 'all_adam_states_step':args.steps,
              'adam_state_entries_each_rank':199, 'native_source_files_verified_each_rank':600,
              'trace_windows_across_ranks':len(captures), 'native_losses':losses,
              'formal_performance_evidence':False, 'collector_drop_count_available':False,
              'full_gradient_optimizer_equivalence_claimed':False}
    with (root/'completion-verification.json').open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
