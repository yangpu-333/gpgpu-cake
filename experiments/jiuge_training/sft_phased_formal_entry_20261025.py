"""Dispatch the exact screened formal entry; no cache advice or new observer."""
import sys
from sft_phased_formal_20261025 import configure
configure()
import sft_phased_acceptance_entry_20261024 as original

if __name__ == '__main__':
    assert sys.argv[sys.argv.index('--sft-pinning-mode')+1] == 'formal'
    original.main()
