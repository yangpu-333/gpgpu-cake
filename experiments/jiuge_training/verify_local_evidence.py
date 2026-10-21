"""Independently review saved CPU tensors and producer hashes; no GPU work."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    case_names = ('native-smoke-256g-01', 'native-profile-256g-01',
                  'native-smoke-s2048-01', 'native-profile-s2048-01')
    cases, inputs, traces = [], [], []
    for name in case_names:
        root = args.results / name
        case = json.loads((root / 'case.json').read_text())
        assert case['exit_code'] == 0
        steps = []
        for rank in range(2):
            entry = json.loads((root / f'rank-{rank}/entry.json').read_text())
            assert entry['status'] == 'completed' and entry['native_files_checked'] == 600
            completed = entry['completed_native_steps']
            assert [row['step'] for row in completed] == list(range(1, len(completed) + 1))
            assert all(row['skipped'] == 0 for row in completed)
            for state in entry['actual_optimizer_state']['optimizers']:
                assert state['adam_state_entries'] == 199
                assert state['min_step'] == state['max_step'] == len(completed)
            steps.append(len(completed))
        assert steps[0] == steps[1]
        cases.append({'case': name, 'steps_per_rank': steps, 'exit_code': 0})
        for folder in sorted(root.glob('window-*')):
            for rank in range(2):
                receipt = json.loads((folder / f'rank-{rank}-capture.json').read_text())
                path = folder / f'rank-{rank}.json.gz'
                assert sha(path) == receipt['trace_sha256']
                value = hashlib.sha256()
                with gzip.open(path, 'rb') as stream:
                    while block := stream.read(1024 * 1024):
                        value.update(block)
                assert value.hexdigest() == receipt['decompressed_json_sha256']
                assert not receipt['launch_ids_without_gpu_kernel']
                assert receipt['recorded_kernel_launches'] == receipt['gpu_kernel_events']
                traces.append({'case': name, 'window': folder.name, 'rank': rank,
                               'compressed_sha256': receipt['trace_sha256'],
                               'decompressed_sha256': value.hexdigest()})
    root = args.results / 'native-smoke-s2048-01'
    rank_tensors = []
    for rank in range(2):
        saved = []
        folder = root / f'rank-{rank}'
        rows = json.loads((folder / 'actual-inputs.json').read_text())
        assert len(rows) == 3
        for row in rows:
            path = folder / row['raw_file']
            assert sha(path) == row['raw_sha256']
            tensors = torch.load(path, map_location='cpu', weights_only=True)
            for key, tensor in tensors.items():
                assert tensor.device.type == 'cpu'
                actual = hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                assert actual == row['fields'][key]['tensor_bytes_sha256']
            tokens, labels, mask = (tensors[key] for key in ('tokens', 'labels', 'loss_mask'))
            assert list(tokens.shape) == [1, 2048]
            assert torch.equal(labels[:, :-1], tokens[:, 1:])
            assert bool((mask == 1).all()) and int(mask.sum()) == 2048
            assert torch.equal(tensors['position_ids'], torch.arange(2048).unsqueeze(0))
            expected = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1)[None, None]
            assert torch.equal(tensors['attention_mask'], expected)
            saved.append(tensors)
            inputs.append({'rank': rank, 'batch': row['batch'], 'raw_sha256': row['raw_sha256'],
                           'all_fields_bytes_verified': True, 'causal_attention_mask_verified': True,
                           'full_document_loss_mask_verified': True, 'next_token_shift_verified': True})
        rank_tensors.append(saved)
    for first, second in zip(*rank_tensors):
        assert first.keys() == second.keys()
        assert all(torch.equal(first[key], second[key]) for key in first)
    result = {'schema': 'jiuge7b-local-numeric-evidence-review-v1', 'cases': cases,
              'trace_count': len(traces), 'traces': traces, 'actual_inputs': inputs,
              'tp_ranks_received_identical_actual_inputs': True,
              'local_cpu_torch_version': torch.__version__, 'gpu_work_started': False,
              'full_optimizer_update_equivalence_tested': False,
              'complete_checkpoint_resume_tested': False, 'formal_performance_passed': False,
              'passed': True}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'passed': True, 'cases': len(cases), 'trace_files_verified': len(traces),
                      'actual_input_batches_verified': len(inputs)}))


if __name__ == '__main__':
    main()
