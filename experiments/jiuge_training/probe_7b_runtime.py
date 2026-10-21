"""CPU-only environment, tokenizer, and optional safe 7B weight metadata check."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys


PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def probe(checkpoint: Path | None, inspect_weights: bool, receipt: Path) -> dict:
    prep = PREP.resolve(strict=True)
    receipt = receipt.resolve()
    if not receipt.is_relative_to(prep):
        raise ValueError('Receipt must remain inside preparation directory')
    if Path(sys.prefix).resolve(strict=True) != (prep / '.venv').resolve(strict=True):
        raise ValueError('Run using the isolated preparation environment')
    sys.dont_write_bytecode = True
    os.environ['HF_HOME'] = str(prep / 'huggingface-cache')
    os.environ['HF_MODULES_CACHE'] = str(prep / 'huggingface-cache' / 'modules')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    result = {
        'status': 'running', 'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'python': sys.version, 'prefix': sys.prefix,
        'gpu_tensors_allocated': False, 'training_performed': False,
        'full_model_instantiated': False, 'packages': {},
    }
    try:
        import torch
        import transformers
        import safetensors
        import sentencepiece
        import accelerate
        import datamodel_code_generator
        import jsonschema
        from datamodel_code_generator import DataModelType
        from datamodel_code_generator.format import PythonVersion
        from datamodel_code_generator.model import get_data_model_types
        from datamodel_code_generator.parser.jsonschema import JsonSchemaParser
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        for name in ('torch', 'triton', 'transformers', 'safetensors', 'sentencepiece',
                     'accelerate', 'datamodel-code-generator', 'jsonschema'):
            result['packages'][name] = metadata.version(name)
        result['torch_import_path'] = torch.__file__
        result['torch_cuda_build'] = torch.version.cuda
        if metadata.version('transformers') != '4.44.0':
            raise ValueError('7B environment must use its pinned Transformers version')
        if not Path(torch.__file__).resolve().is_relative_to(prep / '.venv'):
            raise ValueError('Torch imported from outside isolated environment')
        if checkpoint is not None:
            checkpoint = checkpoint.resolve(strict=True)
            if not checkpoint.is_relative_to(prep) or checkpoint.name != '9G7B_MHA':
                raise ValueError('Checkpoint must be the official 7B preparation directory')
            config = transformers.AutoConfig.from_pretrained(
                str(checkpoint), trust_remote_code=True, local_files_only=True)
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(checkpoint), trust_remote_code=True, local_files_only=True)
            model_class = get_class_from_dynamic_module(
                'modeling_fm9g.FM9GForCausalLM', str(checkpoint), local_files_only=True)
            sample = tokenizer('7B tokenizer readiness test.', add_special_tokens=True)
            chat = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': 'test'}, {'role': 'assistant', 'content': 'ok'}],
                tokenize=False, add_generation_prompt=False)
            result['checkpoint'] = str(checkpoint)
            result['config'] = {name: getattr(config, name) for name in (
                'model_type', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'vocab_size',
                'max_position_embeddings', 'rms_norm_eps', 'rope_theta', 'tie_word_embeddings')}
            result['model_class_imported'] = model_class.__name__
            result['tokenizer'] = {
                'class': type(tokenizer).__name__, 'length': len(tokenizer),
                'eos_token': tokenizer.eos_token, 'eos_token_id': tokenizer.eos_token_id,
                'pad_token_id': tokenizer.pad_token_id,
                'add_bos_token': getattr(tokenizer, 'add_bos_token', None),
                'add_eos_token': getattr(tokenizer, 'add_eos_token', None),
                'sample_input_ids': sample['input_ids'], 'chat_template_sample': chat,
            }
            if inspect_weights:
                weights_file = checkpoint / 'pytorch_model.bin'
                if weights_file.is_symlink() or not weights_file.resolve(strict=True).is_relative_to(checkpoint):
                    raise ValueError('Weight file escapes checkpoint directory')
                tensors = torch.load(str(weights_file), mmap=True, map_location='meta', weights_only=True)
                if not isinstance(tensors, dict):
                    raise ValueError('Official checkpoint is not a tensor mapping')
                if not all(isinstance(value, torch.Tensor) and value.device.type == 'meta'
                           for value in tensors.values()):
                    raise ValueError('Checkpoint entries must all be meta tensors')
                count = sum(value.numel() for value in tensors.values())
                if count != 7668831744 or len(tensors) != 339:
                    raise ValueError('7B checkpoint tensor count or parameter count mismatch')
                result['weights_metadata'] = {
                    'method': 'torch.load(mmap=True,map_location=meta,weights_only=True)',
                    'bytes': weights_file.stat().st_size, 'tensors': len(tensors),
                    'parameter_elements': count,
                    'dtype_counts': dict(collections.Counter(str(value.dtype) for value in tensors.values())),
                    'tensor_shapes': {name: list(value.shape) for name, value in tensors.items()},
                }
        freeze = sorted(f"{item.metadata['Name']}=={item.version}"
                        for item in metadata.distributions() if item.metadata.get('Name'))
        freeze_path = prep / 'runtime-freeze-7b.txt'
        freeze_path.write_text('\n'.join(freeze) + '\n')
        result['freeze_path'] = str(freeze_path)
        result['status'] = 'complete'
    except Exception as exc:
        result['status'] = 'failed'
        result['error'] = repr(exc)
        raise
    finally:
        receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--inspect-weights', action='store_true')
    parser.add_argument('--receipt', type=Path, default=PREP / 'runtime-probe-7b.json')
    args = parser.parse_args()
    result = probe(args.checkpoint, args.inspect_weights, args.receipt)
    print(json.dumps({'status': result['status'], 'packages': result['packages'],
                      'weights_metadata': {k: v for k, v in result.get('weights_metadata', {}).items()
                                           if k != 'tensor_shapes'}}))
