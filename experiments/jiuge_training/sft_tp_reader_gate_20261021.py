"""Check dataset logs under native TP2 reader/broadcast semantics.

Only TP rank0 calls next(data_iterator). The receiver's empty dataset log is
expected. These metadata checks do not claim a new raw receiver-tensor audit.
"""


def check_rank_rows(rank, rows, reader_rows):
    if rank not in (0, 1):
        raise ValueError('Exactly two TP ranks required')
    if len(reader_rows) != 35:
        raise ValueError('The native dataset reader must record all35 batches')
    if [r['requested_sample'] for r in reader_rows] != list(range(35)):
        raise ValueError('Frozen first35 native sampler IDs required')
    for row in reader_rows:
        if row['input_slots'] != 2048 or not 0 < row['supervised_answer_and_EOD_targets'] <= 2048:
            raise ValueError('Actual S2048 answer/EOD supervision counts required')
    if not any(r['supervised_answer_and_EOD_targets'] < 2048 for r in reader_rows):
        raise ValueError('Prompt masking is absent')
    expected = reader_rows if rank == 0 else []
    if rows != expected:
        raise ValueError('Dataset log differs from native TP reader/receiver role')
    return {'rank': rank, 'dataset_reader': rank == 0, 'recorded_dataset_batches': len(rows),
            'metadata_gate_passed': True, 'raw_receiver_tensor_equivalence_claimed': False}
