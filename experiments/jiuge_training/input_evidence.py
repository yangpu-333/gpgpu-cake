"""Copy the first real native batches during admission, outside timing windows."""
import hashlib
import json
from pathlib import Path


def install_input_observer(torch, forward_step, root, rows, limit=3):
    namespace = getattr(forward_step, '__globals__', {})
    original = namespace.get('get_batch')
    if getattr(original, '_jiuge_input_observer', False):
        return
    keys = namespace.get('BATCH_KEYS')
    if not callable(original) or not isinstance(keys, list):
        raise ValueError('The pinned native batch schema is unavailable')

    def observed(*args, **kwargs):
        batch = original(*args, **kwargs)
        if len(rows) < limit:
            if len(batch) != len(keys):
                raise ValueError('Native batch schema length differs')
            tensors = {key: value.detach().cpu().clone() for key, value in zip(keys, batch)
                       if isinstance(value, torch.Tensor)}
            if not {'tokens', 'labels', 'loss_mask', 'position_ids'}.issubset(tensors):
                raise ValueError('Full native CPT input fields required')
            path = Path(root) / f'actual-input-{len(rows) + 1}.pt'
            torch.save(tensors, path)
            tokens, labels, mask = (tensors[key] for key in ('tokens', 'labels', 'loss_mask'))
            record = {'batch': len(rows) + 1, 'raw_file': path.name,
                      'raw_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                      'fields': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                                      'tensor_bytes_sha256': hashlib.sha256(
                                          value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
                                 for key, value in tensors.items()},
                      'loss_mask_all_ones': bool((mask == 1).all()),
                      'valid_loss_tokens': float(mask.sum()),
                      'labels_shift_matches_tokens_inside_sequence': bool(
                          torch.equal(labels[..., :-1], tokens[..., 1:])),
                      'token_min': int(tokens.min()), 'token_max': int(tokens.max()),
                      'label_min': int(labels.min()), 'label_max': int(labels.max()),
                      'admission_only_extra_cpu_copy': True}
            rows.append(record)
            (Path(root) / 'actual-inputs.json').write_text(json.dumps(rows, indent=2) + '\n')
        return batch

    observed._jiuge_input_observer = True
    namespace['get_batch'] = observed
