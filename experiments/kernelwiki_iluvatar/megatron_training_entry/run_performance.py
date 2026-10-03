"""Run three fixed independent formal-loop pairs, alternating process order."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('megatron_root', type=Path)
    args = parser.parse_args()
    contract = json.loads((ROOT/'contract.json').read_text())
    for index, seed in enumerate(contract['performance']['independent_seeds']):
        arms = ('native','optimized') if index % 2 == 0 else ('optimized','native')
        for arm in arms:
            case = '%s-perf-%d'%(arm,seed)
            command = [sys.executable,str(ROOT/'run_case.py'),case,'--mode',arm,
                       '--megatron-root',str(args.megatron_root),'--steps',str(contract['performance']['steps']),
                       '--seed',str(seed)]
            result = subprocess.run(command, capture_output=True, text=True)
            print(case, 'completed' if result.returncode == 0 else 'failed', flush=True)
            if result.returncode:
                print(result.stdout, result.stderr, flush=True)
                return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
