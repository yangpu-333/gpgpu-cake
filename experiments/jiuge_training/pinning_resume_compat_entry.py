"""Apply the checkpoint API adapter only to independent untimed resumes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from checkpoint_load_compat import install

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint-compat-receipt', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    receipt = Path(str(args.checkpoint_compat_receipt).replace('{rank}', os.environ['RANK']))
    if receipt.exists():
        raise ValueError('Immutable compatibility receipt already exists')
    evidence = install()
    evidence['adapter_sha256'] = hashlib.sha256((PREP / 'checkpoint_load_compat.py').read_bytes()).hexdigest()
    evidence['run_returned_successfully'] = False
    sys.argv = [str(PREP / 'pinning_semantics_entry.py'), *remaining]
    try:
        runpy.run_path(str(PREP / 'pinning_semantics_entry.py'), run_name='__main__')
        evidence['run_returned_successfully'] = True
    finally:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open('x') as stream:
            json.dump(evidence, stream, indent=2)


if __name__ == '__main__':
    main()
