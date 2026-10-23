"""One real SFT update for memory admission; never a performance result."""
import json
from pathlib import Path

from sft_moment_direct_zero_entry_20261021 import derive as profile_entry

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source, expected):
    text = profile_entry(source, expected)
    changes = (
        ("if '--capture-windows' not in remaining or '--pinning-arm' not in remaining:",
         "if '--capture-windows' in remaining or '--pinning-arm' not in remaining:"),
        ('Separate SFT profile with accepted pinned repair required',
         'One-update memory admission with accepted pinned repair; no profiler'),
        ('optimizer is None or steps != 35', 'optimizer is None or steps != 1'),
        ('All35 real updates and a live optimizer required',
         'Exactly one real update and a live optimizer required'),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Frozen first-update derivation differs: ' + before)
        text = text.replace(before, after)
    return text


def main():
    files = json.loads((PREP/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']
    text = derive((PREP/'sft_numa_entry_20261021.py').read_bytes(), files['sft_numa_entry_20261021.py'])
    exec(compile(text, '<direct-zero-one-update-entry>', 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
