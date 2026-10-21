"""Build answer-only labels aligned with the existing, frozen CPT token stream.

CPU only. The full text, BOS/EOD, document order, and native sampler stay the
same as CPT; a parallel IndexedDataset holds the per-target supervision mask.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
RAW_SHA = '6fcb98c6d79691d1f9a88ef513335da9124e7fdeef5103343f5d9f9a6e8f4903'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def prefix_text(row):
    inputs = '### 输入：\n' + row['input'] + '\n' if row['input'].strip() else ''
    return '### 指令：\n' + row['instruction'] + '\n' + inputs + '### 回答：\n'


def answer_start(full, prefix):
    common = 0
    for left, right in zip(full, prefix):
        if left != right:
            break
        common += 1
    if common == len(prefix):
        return common, False
    # A boundary token may merge the final prompt separator with answer text.
    # Exclude that whole token instead of supervising any prompt characters.
    if len(prefix) - common != 1:
        raise ValueError('Tokenizer changed more than the single boundary token')
    return common + 1, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=int, default=0)
    args = parser.parse_args()
    os.environ.update(USE_TF='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from megatron.core.datasets.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
    source = PREP / 'alpaca_gpt4_data_zh.sft-source.json'
    if sha(source) != RAW_SHA:
        raise ValueError('Frozen source dataset differs')
    rows = json.loads(source.read_text())
    if len(rows) != 48818:
        raise ValueError('Expected all 48,818 source records')
    tokens_prefix = PREP / 'datasets/alpaca-cpt-v1_text_document'
    dataset = IndexedDataset(str(tokens_prefix), multimodal=False, mmap=True)
    tokenizer = AutoTokenizer.from_pretrained(str(PREP / 'checkpoints/9G7B_MHA'),
                                              trust_remote_code=True, local_files_only=True)
    target = PREP / 'datasets/alpaca-sft-answer-v1'
    mask_prefix = Path(str(target) + '_loss_mask')
    receipt = Path(str(target) + '-receipt.json')
    builder = None
    if not args.probe:
        if any(Path(str(prefix) + suffix).exists() for prefix in (target, mask_prefix)
               for suffix in ('.bin', '.idx')) or receipt.exists():
            raise ValueError('Immutable SFT output already exists')
        builder = IndexedDatasetBuilder(str(mask_prefix) + '.bin', dtype=np.int32)
    count = min(args.probe, len(rows)) if args.probe else len(rows)
    total = supervised = mixed = 0
    evidence = []
    journal = None if args.probe else Path(str(target) + '-documents.jsonl').open('x')
    try:
        for index, row in enumerate(rows[:count]):
            prefix = prefix_text(row)
            full = tokenizer.encode(prefix + row['output'], add_special_tokens=True)
            full.append(tokenizer.eos_token_id)
            existing = dataset[index]
            if full != existing.tolist():
                raise ValueError(f'CPT token bytes differ from source/tokenizer at document {index}')
            prompt = tokenizer.encode(prefix, add_special_tokens=True)
            start, boundary = answer_start(full, prompt)
            mask = np.zeros(len(full), dtype=np.int32)
            mask[start:] = 1
            if not 0 < start < len(full) or full[-1] != 73440:
                raise ValueError('Nonempty prompt and terminal answer EOD required')
            if builder:
                builder.add_item(torch.from_numpy(mask))
                builder.end_document()
            item = {'document': index, 'tokens': len(full), 'supervision_starts_at_token': start,
                    'supervised_tokens_including_EOD': int(mask.sum()),
                    'mixed_boundary_token_excluded': boundary,
                    'tokens_int32_sha256': hashlib.sha256(np.asarray(full, dtype=np.int32).tobytes()).hexdigest(),
                    'mask_int32_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
            if journal:
                journal.write(json.dumps(item) + '\n')
            if len(evidence) < 5:
                evidence.append(item)
            total += len(full)
            supervised += int(mask.sum())
            mixed += boundary
            if index and index % 5000 == 0:
                print(json.dumps({'documents_complete': index, 'cuda_initialized': torch.cuda.is_initialized()}), flush=True)
    finally:
        if journal:
            journal.close()
    if builder:
        builder.finalize(str(mask_prefix) + '.idx')
        rebuilt = IndexedDataset(str(mask_prefix), multimodal=False, mmap=True)
        if not np.array_equal(dataset.sequence_lengths, rebuilt.sequence_lengths):
            raise ValueError('Target mask document lengths differ from token lengths')
        for suffix in ('.bin', '.idx'):
            os.link(str(tokens_prefix) + suffix, str(target) + suffix)
    record = {'schema': 'jiuge7b-answer-only-sft-data-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'probe_only': bool(args.probe),
              'documents': count, 'input_tokens_including_BOS_EOD': total,
              'supervised_answer_and_EOD_tokens': supervised, 'masked_prompt_tokens': total - supervised,
              'mixed_boundary_tokens_conservatively_excluded': mixed,
              'original_sha256': RAW_SHA, 'source_documents_tokenized_and_checked': count,
              'tokens_bitwise_identical_to_CPT': True, 'all_source_records_retained': count == len(rows),
              'objective': 'full_parameter_supervised_next_token_loss_on_answer_and_EOD_only',
              'attention': 'same dense causal packed-document layout as native CPT; no inter-document reset',
              'training_entry': 'native pretrain_gpt.py plus external answer-mask GPTDataset subclass',
              'native_packed_sft_flag_used': False, 'chat_template_applied': False,
              'loss_mask_alignment': 'stored per text token; shifted by one together with next-token labels',
              'EOD': 73440, 'examples': evidence,
              'processing_code_sha256': sha(__file__), 'cuda_initialized': torch.cuda.is_initialized(),
              'passed': total > supervised > 0 and not torch.cuda.is_initialized()}
    if builder:
        record['files_sha256'] = {str(prefix) + suffix: sha(str(prefix) + suffix)
                                  for prefix in (target, mask_prefix) for suffix in ('.bin', '.idx')}
        record['document_journal_sha256'] = sha(str(target) + '-documents.jsonl')
        with receipt.open('x') as stream:
            json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
