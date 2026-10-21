"""Pin the requested public Alpaca data and serialize whole-text CPT documents."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REPOSITORY = 'Instruction-Tuning-with-GPT-4/GPT-4-LLM'
DATA_PATH = 'data/alpaca_gpt4_data_zh.json'
TEMPLATE = '### 指令：\n{instruction}\n{input_section}### 回答：\n{output}'


def document(row):
    for key in ('instruction', 'input', 'output'):
        if not isinstance(row.get(key), str):
            raise ValueError('Expected string field: ' + key)
    if not row['instruction'].strip():
        raise ValueError('Empty instruction')
    input_section = '### 输入：\n' + row['input'] + '\n' if row['input'].strip() else ''
    return TEMPLATE.format(instruction=row['instruction'], input_section=input_section, output=row['output'])


def get(url, limit):
    request = urllib.request.Request(url, headers={'User-Agent': 'Jiuge-Training-Preparation'})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Source exceeds configured download bound')
    return data


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    names = ('alpaca_gpt4_data_zh.json', 'alpaca_gpt4_data_zh.cpt.jsonl', 'dataset-receipt.json')
    if any((root / name).exists() for name in names):
        raise ValueError('New dataset receipt directory required; previous records are preserved')
    revision = json.loads(get('https://api.github.com/repos/' + REPOSITORY + '/commits/main', 1024 * 1024))['sha']
    url = 'https://raw.githubusercontent.com/' + REPOSITORY + '/' + revision + '/' + DATA_PATH
    raw = get(url, 128 * 1024 * 1024)
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected nonempty JSON record array')
    raw_path, output = root / names[0], root / names[1]
    raw_path.write_bytes(raw)
    input_count = empty_output_count = 0
    output_hash = hashlib.sha256()
    with output.open('xb') as stream:
        for row in rows:
            text = document(row)
            input_count += bool(row['input'].strip())
            empty_output_count += not bool(row['output'].strip())
            encoded = (json.dumps({'text': text}, ensure_ascii=False) + '\n').encode('utf-8')
            output_hash.update(encoded)
            stream.write(encoded)
    record = {'schema': 'jiuge-alpaca-cpt-data-v1', 'utc': datetime.now(timezone.utc).isoformat(),
              'repository': 'https://github.com/' + REPOSITORY, 'revision': revision,
              'original_requested_path': DATA_PATH, 'download_url': url,
              'original_bytes': len(raw), 'original_sha256': hashlib.sha256(raw).hexdigest(),
              'records': len(rows), 'records_with_input': input_count,
              'records_with_empty_output': empty_output_count,
              'prepared_path': str(output.resolve()), 'prepared_bytes': output.stat().st_size,
              'prepared_sha256': output_hash.hexdigest(),
              'processing_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'text_template': TEMPLATE, 'record_order': 'unchanged source array order',
              'objective': 'continued_pretraining_next_token_on_entire_document',
              'answer_only_loss_mask': False, 'chat_template_applied': False,
              'tokenizer_and_EOD_not_yet_applied': True,
              'packing_sequence_length_and_training_precision_not_yet_frozen': True,
              'training_started': False}
    (root / names[2]).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
