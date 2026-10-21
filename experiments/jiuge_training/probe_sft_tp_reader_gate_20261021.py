"""CPU-only TP metadata regression; no new training equivalence claim."""
from copy import deepcopy
import json
from sft_tp_reader_gate_20261021 import check_rank_rows


def main():
    rows = [{'requested_sample': i, 'input_slots': 2048,
             'supervised_answer_and_EOD_targets': 1700+i} for i in range(35)]
    checks = []
    check_rank_rows(0, rows, rows)
    check_rank_rows(1, [], rows)
    checks.append('Native TP2 reader35/receiver0 dataset-log roles accepted')
    fixtures = [(0, rows[:-1], rows, 'Missing reader log rejected'),
                (1, rows, rows, 'Receiver unexpectedly consuming its own dataset rejected')]
    wrong = deepcopy(rows)
    wrong[5]['supervised_answer_and_EOD_targets'] = 0
    fixtures.append((0, wrong, wrong, 'Empty supervised batch rejected'))
    wrong = deepcopy(rows)
    wrong[5]['requested_sample'] = 100
    fixtures.append((0, wrong, wrong, 'Different sampler IDs rejected'))
    fixtures.append((0, [dict(x, input_slots=128) for x in rows], rows, 'Reader metadata different from paired reference rejected'))
    for rank, actual, reference, label in fixtures:
        try:
            check_rank_rows(rank, actual, reference)
        except ValueError:
            checks.append(label)
        else:
            raise AssertionError(label)
    print(json.dumps({'schema': 'jiuge7b-sft-TP-reader-metadata-regression-v1',
                      'passed': True, 'checks': checks, 'GPU_used': False,
                      'raw_receiver_tensor_equivalence_claimed': False,
                      'formal_performance_evidence': False}, indent=2))


if __name__ == '__main__':
    main()
