"""Official FP32 forward reference from real CPT documents; no training."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HOME=str(PREP/'huggingface-cache'), HF_MODULES_CACHE=str(PREP/'huggingface-cache/modules'),
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    checkpoint = PREP/'checkpoints/9G7B_MHA'
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True,
                    torch_dtype=torch.float32, low_cpu_mem_usage=True, attn_implementation='eager', device_map={'':0})
    model.eval()
    files, rows = {}, []
    with (PREP/'alpaca_gpt4_data_zh.cpt.jsonl').open() as source, torch.no_grad():
        for index, maximum in enumerate((32,64,128)):
            text = json.loads(next(source))['text']
            ids = tokenizer.encode(text, add_special_tokens=True) + [tokenizer.eos_token_id]
            tokens = torch.tensor([ids[:maximum]], dtype=torch.long, device='cuda')
            logits = model(tokens, use_cache=False).logits
            loss = torch.nn.functional.cross_entropy(logits[:,:-1].float().flatten(0,1),tokens[:,1:].flatten())
            path = args.output/f'case-{index}.pt'
            torch.save({'tokens':tokens.cpu(), 'logits':logits.cpu(), 'loss':loss.cpu()},path)
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            rows.append({'case':index, 'sequence':tokens.shape[1], 'loss':loss.item(), 'logits_dtype':str(logits.dtype)})
            print(json.dumps(rows[-1]),flush=True)
    record = {'schema':'jiuge7b-official-forward-reference-v1','utc':datetime.now(timezone.utc).isoformat(),
              'cases':rows,'files_sha256':files,'elapsed_seconds':time.time()-started,
              'parameter_elements':sum(p.numel() for p in model.parameters()),'dtype':'FP32 from official BF16 weights',
              'tf32':False,'backend':'official FM9G eager attention','training_or_optimizer_update':False,
              'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
