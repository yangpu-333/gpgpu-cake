"""Versioned observer compatibility only; original one-update math unchanged."""
from pathlib import Path
import runpy
from sft_memory_snapshot_compat_20261023 import install

if __name__ == '__main__':
    install()
    runpy.run_path(str(Path(__file__).parent/'sft_direct_zero_first_step_entry_20261023.py'),run_name='__main__')
