"""Preserve the accepted semantic/resume checks with an explicit SFT mask gate."""
import argparse
import hashlib
import json
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--resume-campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    text = (PREP/'verify_pinning_semantics.py').read_text()
    before = "if not item['loss_mask_all_ones'] or not item['labels_shift_matches_tokens_inside_sequence'] or item['valid_loss_tokens'] != 2048:"
    after = "if item['loss_mask_all_ones'] or not item['labels_shift_matches_tokens_inside_sequence'] or not 0 < item['valid_loss_tokens'] < 2048:"
    if text.count(before) != 1 or text.count('Not the fixed full-document CPT input contract') != 1:
        raise ValueError('Existing semantic verifier changed')
    text = text.replace(before,after).replace('Not the fixed full-document CPT input contract','Actual answer-only SFT mask or shift differs')
    namespace = {'__name__':'sft_semantics_verifier'}
    exec(compile(text,str(PREP/'verify_sft_pinning_semantics.generated.py'),'exec'),namespace)
    result = namespace['verify'](args.root,args.resume_campaign)
    result.update(objective='answer_only_SFT',sft_mask_gate=True,
                  sft_data_receipt_sha256=hashlib.sha256((PREP/'datasets/alpaca-sft-answer-v1-receipt.json').read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
