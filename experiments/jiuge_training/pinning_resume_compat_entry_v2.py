"""Versioned resume-only compatibility; preserve the failed first attempt."""
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
    sys.path.insert(0, str(PREP / 'megatron-native-5be9626'))
    evidence = install()
    from checkpoint_hdo_load_compat import install as install_hdo
    install_hdo(evidence)
    evidence['adapter_sha256'] = {name: hashlib.sha256((PREP / name).read_bytes()).hexdigest()
        for name in ('checkpoint_load_compat.py', 'checkpoint_hdo_load_compat.py')}
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
