"""New six-fresh-case screen. Never resume or reuse the OOM-interrupted screen."""
import hashlib
import json
from pathlib import Path
from continue_sft_moment_screen_20261021 import derive as previous,moment_gate as previous_gate

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def moment_gate(folder,rank,arm,binding):
    previous_gate(folder,rank,arm,binding)
    if arm=='local-moments':
        events=json.loads((folder/f'sft-local-moments-rank-{rank}.json').read_text())['moment_events']
        if len(events)!=398 or any(e['initialization_path']!='direct_local_zero'
                                  or e['native_source_moment_created'] or e['copied_source_moment_bytes']!=0
                                  for e in events):
            raise ValueError('Every fresh final moment must be created directly; no native source copy')
        progress=[json.loads(line) for line in (folder/f'moment-initialization-memory-rank-{rank}.jsonl').read_text().splitlines()]
        if progress[-1].get('created_moment_objects')!=398 or progress[-1]['copied_source_moment_bytes']!=0:
            raise ValueError('Complete initialization-memory evidence required')


def derive(source):
    text=previous(source)
    changes=[('sft-moment-screen-20261021-01','sft-moment-direct-zero-screen-20261021-01'),
             ('sft-moment-screen-approved-20261021.json','sft-moment-direct-zero-approved-20261021.json'),
             ('sft-moment-runtime-','sft-moment-direct-zero-runtime-'),
             ('sft-moment-screen-20261021-','sft-moment-direct-zero-20261021-'),
             ('run_sft_moment_case_20261021.py','run_sft_moment_direct_zero_case_20261021.py'),
             ('sft-moment-screen-key-results-20261021.zip','sft-moment-direct-zero-key-results-20261021.zip')]
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact new screen derivation required: '+before)
        text=text.replace(before,after)
    return text


if __name__=='__main__':
    files=json.loads((PREP/'sft-moment-direct-zero-source-manifest-v3-20261021.json').read_text())['files']
    source=(PREP/'continue_sft_numa_v2_20261021.py').read_bytes()
    if hashlib.sha256(source).hexdigest()!=files['continue_sft_numa_v2_20261021.py']:
        raise ValueError('Frozen original controller differs')
    exec(compile(derive(source),'<direct-zero-six-fresh-SFT-cases>','exec'),
         {'__name__':'__main__','__file__':__file__,'moment_gate':moment_gate})
