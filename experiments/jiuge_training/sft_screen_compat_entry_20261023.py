"""Same optional-peak observer for every35-update screen arm."""
from pathlib import Path
import runpy
from sft_memory_snapshot_compat_20261023 import install

if __name__ == '__main__':
    install()
    runpy.run_path(str(Path(__file__).parent/'sft_moment_direct_zero_entry_20261021.py'),run_name='__main__')
