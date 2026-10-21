"""Verify staged weight files and optionally run a CPU/meta loading check.

The runtime check creates no real parameter storage and performs no forward,
backward, GPU math, or training. It is not a model conversion equivalence test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import struct
import tempfile
import traceback

from static_model_audit import audit


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read_header(path):
    with path.open('rb') as stream:
        encoded_length = stream.read(8)
        if len(encoded_length) != 8:
            raise ValueError('Truncated safetensors length')
        length, = struct.unpack('<Q', encoded_length)
        if not 2 <= length <= 16 * 1024 * 1024:
            raise ValueError('Unexpected safetensors header size')
        encoded = stream.read(length)
        if len(encoded) != length:
            raise ValueError('Truncated safetensors header')
    header = json.loads(encoded)
    intervals = []
    for name, item in header.items():
        if name == '__metadata__':
            continue
        begin, end = item['data_offsets']
        if item['dtype'] != 'BF16' or end - begin != math.prod(item['shape']) * 2:
            raise ValueError('Unexpected dtype or byte span: ' + name)
        if begin < 0 or end < begin:
            raise ValueError('Invalid data offset: ' + name)
        intervals.append((begin, end))
    position = 0
    for begin, end in sorted(intervals):
        if begin != position:
            raise ValueError('Overlapping or noncontiguous tensor data')
        position = end
    if 8 + length + position != path.stat().st_size:
        raise ValueError('Shard payload length disagrees with tensor offsets')
    return header


def runtime_check(root, headers):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    before = torch.cuda.is_initialized()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=True)
    examples = []
    for text in ('九格模型准备验证。', 'Jiuge checkpoint preparation.'):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if not tokens or any(token < 0 or token >= config.vocab_size for token in tokens):
            raise ValueError('Tokenizer generated invalid token IDs')
        examples.append({'text': text, 'token_ids': tokens,
                         'decoded': tokenizer.decode(tokens, skip_special_tokens=True)})
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                               attn_implementation='eager', torch_dtype=torch.bfloat16)
    state = model.state_dict()
    expected = {key: item for header in headers.values() for key, item in header.items()
                if key != '__metadata__'}
    mismatches = []
    if set(state) != set(expected):
        mismatches.append({'kind': 'key_set', 'missing': sorted(set(expected) - set(state)),
                           'extra': sorted(set(state) - set(expected))})
    for key in set(state) & set(expected):
        if list(state[key].shape) != expected[key]['shape']:
            mismatches.append({'kind': 'shape', 'name': key})
        if state[key].device.type != 'meta':
            mismatches.append({'kind': 'real_storage_allocated', 'name': key})
    after = torch.cuda.is_initialized()
    return {'passed': not mismatches and not before and not after,
            'packages': {name: importlib.metadata.version(name) for name in
                         ('torch', 'triton', 'transformers', 'safetensors', 'sentencepiece', 'accelerate')},
            'tokenizer_class': type(tokenizer).__name__, 'tokenizer_length': len(tokenizer),
            'tokenizer_examples': examples, 'config_class': type(config).__name__,
            'model_class': type(model).__name__, 'model_parameter_elements': sum(p.numel() for p in model.parameters()),
            'state_keys_checked': len(state), 'mismatches': mismatches,
            'cuda_initialized_before': before, 'cuda_initialized_after': after,
            'official_model_code_executed_for_meta_construction': True,
            'parameter_storage': 'meta only', 'forward_backward_or_training_performed': False,
            'megatron_conversion_equivalence_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging-receipt', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime', action='store_true')
    args = parser.parse_args()
    staging = json.loads(args.staging_receipt.read_text())
    if staging.get('status') != 'complete' or staging.get('file_count') != 15:
        raise ValueError('Complete 15-file checkpoint required')
    root = Path(staging['checkpoint_directory']).resolve()
    errors, hashes, headers = [], {}, {}
    for key, item in staging['files'].items():
        path = Path(item['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Receipt path escaped checkpoint directory')
        actual = digest(path)
        hashes[path.name] = actual
        if actual != item['sha256'] or path.stat().st_size != item['bytes']:
            errors.append('File differs from staged receipt: ' + key)
        if path.suffix == '.safetensors':
            headers[path.name] = read_header(path)
        print(json.dumps({'file_rechecked': path.name, 'sha256': actual}), flush=True)
    index = json.loads((root / 'model.safetensors.index.json').read_text())
    actual_map = {key: name for name, header in headers.items() for key in header if key != '__metadata__'}
    if actual_map != index['weight_map']:
        errors.append('Shard index and actual tensor headers disagree')
    actual_tensor_bytes = sum(math.prod(item['shape']) * 2 for header in headers.values()
                              for name, item in header.items() if name != '__metadata__')
    if index['metadata']['total_size'] != actual_tensor_bytes:
        errors.append('Index tensor byte count disagrees with all shard headers')
    with tempfile.TemporaryDirectory(prefix='header-audit-', dir=root.parent) as folder:
        temporary = Path(folder)
        for name in ('config.json', 'configuration_fm9g.py', 'modeling_fm9g.py'):
            shutil.copyfile(root / name, temporary / name)
        for name, header in headers.items():
            (temporary / (name + '.header.json')).write_text(json.dumps(header), encoding='utf-8')
        static = audit(temporary)
    if not static['public_metadata_consistent']:
        errors.extend(static['errors'])
    record = {'schema': 'jiuge-staged-checkpoint-verification-v1',
              'utc': datetime.now(timezone.utc).isoformat(), 'directory': str(root),
              'staging_receipt_sha256': digest(args.staging_receipt),
              'all_files_independently_rehashed': True, 'file_count': len(hashes),
              'file_sha256': hashes, 'headers_and_index_match': not errors,
              'errors': errors, 'static_model_audit': static,
              'full_tensor_payload_loaded_into_memory': False,
              'gpu_math_or_training_performed': False}
    if args.runtime:
        try:
            record['runtime_check'] = runtime_check(root, headers)
        except Exception as exc:
            record['runtime_check'] = {'passed': False, 'error_type': type(exc).__name__,
                                       'traceback': traceback.format_exc()}
    record['passed'] = not errors and (not args.runtime or record['runtime_check']['passed'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'passed': record['passed'], 'output': str(args.output)}))
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
