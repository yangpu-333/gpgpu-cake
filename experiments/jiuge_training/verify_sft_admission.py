"""Reconstruct real SFT input batches from source documents, outside profiling."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root', type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    sys.path.insert(0, str(PREP/'megatron-native-5be9626'))
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    root = args.case_root
    case = json.loads((root/'case.json').read_text())
    if case['exit_code'] != 0:
        raise ValueError('Native SFT training must complete successfully')
    entries = [json.loads((root/f'rank-{rank}/entry.json').read_text()) for rank in range(2)]
    for entry in entries:
        if entry['status'] != 'completed' or entry['native_files_checked'] != 600:
            raise ValueError('Frozen native entry/source validation failed')
        if entry['completed_native_steps'] != [{'step': i, 'skipped': 0} for i in (1, 2, 3)]:
            raise ValueError('Exactly three non-skipped native updates required')
        optimizers = entry['actual_optimizer_state']['optimizers']
        if sum(x['adam_state_entries'] for x in optimizers) != 199:
            raise ValueError('Full-parameter AdamW state coverage differs')
        if any(x['min_step'] != 3 or x['max_step'] != 3 for x in optimizers if x['adam_state_entries']):
            raise ValueError('Actual optimizer steps differ from 3')
    source = PREP/'alpaca_gpt4_data_zh.sft-source.json'
    if hashlib.sha256(source.read_bytes()).hexdigest() != '6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903':
        raise ValueError('Original data hash differs')
    rows = json.loads(source.read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(PREP/'checkpoints/9G7B_MHA'),
                                              trust_remote_code=True, local_files_only=True)
    data = IndexedDataset(str(PREP/'datasets/alpaca-cpt-v1_text_document'), multimodal=False, mmap=True)
    provenance = json.loads((root/'sft-samples-rank-0.json').read_text())
    if len(provenance) != 3:
        raise ValueError('Exactly three real native sampled batches required')
    checks = []
    for ordinal, sample in enumerate(provenance, 1):
        parts, masks = [], []
        ids = sample['document_ids']
        for number, document_id in enumerate(ids):
            row = rows[document_id]
            inputs = ('### 输入：\n' + row['input'] + '\n') if row['input'].strip() else ''
            prompt = '### 指令：\n' + row['instruction'] + '\n' + inputs + '### 回答：\n'
            prefix = tokenizer.encode(prompt, add_special_tokens=True)
            full = tokenizer.encode(prompt + row['output'], add_special_tokens=True) + [73440]
            if full != data[document_id].tolist():
                raise ValueError('Actual document does not match original source')
            common = next((j for j, (a, b) in enumerate(zip(prefix, full)) if a != b), len(prefix))
            if common == len(prefix):
                start = len(prefix)
            elif len(prefix) - common == 1:
                start = common + 1
            else:
                raise ValueError('Ambiguous token boundary in source supervision')
            answer = np.asarray([int(j >= start) for j in range(len(full))], dtype=np.float32)
            left = sample['start_document_index_and_offset'][1] if number == 0 else 0
            right = sample['end_document_index_and_offset'][1] + 1 if number == len(ids)-1 else len(full)
            parts.extend(full[left:right])
            masks.extend(answer[left:right].tolist())
        length = sample['sequence_length']
        if len(parts) != length + 1:
            raise ValueError('Independent source reconstruction length differs')
        expected = {'tokens': torch.tensor(parts[:-1], dtype=torch.long).unsqueeze(0),
                    'labels': torch.tensor(parts[1:], dtype=torch.long).unsqueeze(0),
                    'loss_mask': torch.tensor(masks[1:], dtype=torch.float32).unsqueeze(0),
                    'position_ids': torch.arange(length, dtype=torch.long).unsqueeze(0)}
        if not 0 < expected['loss_mask'].sum().item() < length:
            raise ValueError('Both supervised answers and masked prompts required')
        rank_checks = []
        previous = None
        for rank in range(2):
            path = root/f'rank-{rank}/actual-input-{ordinal}.pt'
            stored = json.loads((root/f'rank-{rank}/actual-inputs.json').read_text())[ordinal-1]
            if hashlib.sha256(path.read_bytes()).hexdigest() != stored['raw_sha256']:
                raise ValueError('Captured raw input SHA differs')
            actual = torch.load(path, map_location='cpu', weights_only=True)
            equal = {key: torch.equal(actual[key], value) for key, value in expected.items()}
            mask = actual.get('attention_mask')
            causal = (mask is not None and torch.equal(mask[0, 0],
                      torch.triu(torch.ones((length,length), dtype=torch.bool), diagonal=1)))
            tp_same = previous is None or all(torch.equal(previous[key], actual[key]) for key in expected)
            if not all(equal.values()) or not causal or not tp_same:
                raise ValueError('Actual SFT batch differs from independent source/mask reconstruction')
            rank_checks.append({'rank': rank, 'source_fields_equal': equal, 'causal_attention': causal,
                                'TP_broadcast_identical': tp_same, 'raw_sha256': stored['raw_sha256']})
            previous = actual
        checks.append({'batch': ordinal, 'source_document_ids': ids,
                       'input_slots': length, 'supervised_answer_and_EOD_targets': int(expected['loss_mask'].sum()),
                       'masked_prompt_targets': int((expected['loss_mask'] == 0).sum()), 'ranks': rank_checks})
    log = (root/'stdout.log').read_text()
    losses = [float(x) for x in re.findall(r'lm loss:\s*([\d.E+\-]+)', log)]
    if len(losses) != 3 or not all(np.isfinite(x) for x in losses):
        raise ValueError('Three finite native training loss values required')
    result = {'schema': 'jiuge7b-sft-real-batch-admission-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'case': case['case'],
              'passed': True, 'native_files_verified_each_rank': 600, 'optimizer_updates_each_rank': 3,
              'adam_state_entries_each_rank': 199, 'native_losses': losses, 'batches': checks,
              'full_parameter_training': True, 'all_source_tokens_equal_to_CPT': True,
              'answer_only_supervision_source_reconstructed': True,
              'all_model_gradient_and_optimizer_tensors_numerically_audited': False,
              'scope': 'native SFT task/real-input/update admission; no optimizer candidate or formal speedup claim',
              'cuda_initialized': torch.cuda.is_initialized()}
    with (root/'sft-admission-verification.json').open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
