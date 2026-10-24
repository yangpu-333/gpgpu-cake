"""Reuse live guarded native CPU checks, then acceptance-specific wiring checks."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from sft_phased_acceptance_20261024 import PREP, replace_exact


def main():
    text = replace_exact((PREP/'probe_sft_runtime_phased_v2_20261023.py').read_text(), [
        ('sft-phased-moments-approved-20261023-07.json', 'sft-phased-acceptance-approved-20261024-01.json'),
    ])
    exec(compile(text, '<same-live-eight-GiB-and-native-fused-fixtures>', 'exec'),
         {'__name__':'__main__','__file__':__file__})
    result = subprocess.check_output([sys.executable,str(PREP/'check_sft_phased_acceptance_20261024.py')],text=True)
    checks = json.loads(result.strip().splitlines()[-1])
    if not checks['passed'] or checks['GPU_used']:
        raise ValueError('Acceptance CPU dispatch gates failed')
    root = PREP/('sft-moment-direct-zero-runtime-'+socket.gethostname())
    with (root/'acceptance-wiring.json').open('x') as stream:
        json.dump(checks,stream,indent=2)


if __name__ == '__main__':
    main()
