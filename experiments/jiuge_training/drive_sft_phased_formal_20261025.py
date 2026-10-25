"""One locked bounded-key observer; never starts a GPU experiment."""
import subprocess
import sys
from pathlib import Path
import drive_sft_phased_acceptance_20261024 as previous
from pinning_key_results import KEY_SUFFIXES

BASE = Path(__file__).resolve().parent
previous.LOCAL = BASE/'results/sft-phased-formal-driver-20261025-01'
previous.ROOT = previous.PREP+'/sft-phased-formal-20261025-01'
KEY_SUFFIXES.add('.sh')

if __name__ == '__main__':
    code = previous.main()
    if code == 0:
        result = subprocess.run([sys.executable, str(BASE/'review_sft_phased_formal_20261025.py')],
                                cwd=BASE, check=False)
        if result.returncode:
            raise SystemExit(result.returncode)
    raise SystemExit(code)
