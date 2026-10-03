"""Record bounded initial probes for the pinned formal Megatron entry point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--megatron-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    probes = {
        'minimum-allocation': [sys.executable, '-c',
            "import torch; x=torch.zeros(1).cuda(); torch.cuda.synchronize(); print(torch.__version__,torch.cuda.get_device_name(0),x.cpu().tolist())"],
        'native-pretrain-help': [sys.executable, '-u', str(args.megatron_root / 'pretrain_gpt.py'), '--help'],
    }
    records = []
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    for name, command in probes.items():
        record = {'name': name, 'command': command, 'started_utc': datetime.now(timezone.utc).isoformat()}
        with (args.output_dir / (name + '.stdout')).open('w') as out, (args.output_dir / (name + '.stderr')).open('w') as err:
            try:
                record['exit_code'] = subprocess.run(command, stdout=out, stderr=err,
                    env=env, cwd=args.megatron_root, timeout=60).returncode
            except subprocess.TimeoutExpired:
                record.update(exit_code=124, error='60 second timeout')
        records.append(record)
    (args.output_dir / 'commands.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
