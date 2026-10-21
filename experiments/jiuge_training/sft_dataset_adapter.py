"""Answer-only loss mask adapter; preserve frozen native model and sampler math."""
from types import SimpleNamespace
import json
import os
from pathlib import Path


def install():
    import numpy as np
    import torch
    import megatron.core.datasets.gpt_dataset as module
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    original = module.GPTDataset

    class AnswerMaskedGPTDataset(original):
        sampled_supervision_rows = []

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.answer_masks = IndexedDataset(self.dataset.path_prefix + '_loss_mask',
                                               multimodal=False, mmap=True)
            if not np.array_equal(self.dataset.sequence_lengths, self.answer_masks.sequence_lengths):
                raise ValueError('SFT mask and token document lengths differ')
            self.admission_samples = []

        def __getitem__(self, idx):
            result = super().__getitem__(idx)
            proxy = SimpleNamespace(dataset=self.answer_masks, shuffle_index=self.shuffle_index,
                                    sample_index=self.sample_index, document_index=self.document_index,
                                    config=self.config, _pad_token_id=0)
            values, _, _ = original._query_document_sample_shuffle_indices(proxy, 0 if idx is None else idx)
            aligned = values[1:] if self.config.add_extra_token_to_sequence else np.concatenate((values[1:], [0]))
            if aligned.shape != tuple(result['labels'].shape) or not np.isin(aligned, [0, 1]).all():
                raise ValueError('SFT label-mask alignment or binary mask failed')
            result['loss_mask'] = result['loss_mask'] * torch.from_numpy(aligned.copy()).float()
            if idx is None:
                result['loss_mask'].zero_()
            elif result['loss_mask'].sum().item() == 0:
                raise ValueError('SFT sampled window has no supervised answer tokens')
            if idx is not None:
                type(self).sampled_supervision_rows.append({'requested_sample': int(idx),
                    'input_slots': int(result['tokens'].numel()),
                    'supervised_answer_and_EOD_targets': int(result['loss_mask'].sum().item())})
            evidence_root = os.environ.get('JIUGE_SFT_ADMISSION_ROOT')
            if evidence_root and idx is not None and len(self.admission_samples) < 3:
                sample = int(self.shuffle_index[idx])
                begin, end = self.sample_index[sample], self.sample_index[sample + 1]
                self.admission_samples.append({'requested_sample': int(idx), 'shuffled_sample': sample,
                    'start_document_index_and_offset': [int(x) for x in begin],
                    'end_document_index_and_offset': [int(x) for x in end],
                    'document_ids': [int(x) for x in self.document_index[int(begin[0]):int(end[0]) + 1]],
                    'supervised_targets': int(result['loss_mask'].sum().item()),
                    'sequence_length': int(result['tokens'].numel())})
                path = Path(evidence_root) / ('sft-samples-rank-' + os.environ.get('RANK', '0') + '.json')
                path.write_text(json.dumps(self.admission_samples, indent=2) + '\n')
            return result

    module.GPTDataset = AnswerMaskedGPTDataset
    return AnswerMaskedGPTDataset
