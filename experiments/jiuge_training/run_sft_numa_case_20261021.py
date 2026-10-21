"""Versioned affinity wrapper around the frozen, accepted SFT trace runner."""
import hashlib
from pathlib import Path

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
EXPECTED = '2eaf0abe1ac7a2983a02f7047c44f4d442b150ef06ef4abb14a01b070d088e0a'


def derive(source):
    if hashlib.sha256(source).hexdigest() != EXPECTED:
        raise ValueError('Frozen Oct20 SFT trace runner changed')
    text = source.decode()
    replacements = [
        ('def main():', 'def generated_main():'),
        ('if __name__ == \'__main__\':\n    main()', "if __name__ == '__main__':\n    generated_main()"),
        ('str(PREP/\'sft_trace_entry_20261020.py\')', "str(PREP/'sft_numa_entry_20261021.py')"),
        ("parser.add_argument('--pinning-arm',choices=('native','optimized'),required=True)",
         "parser.add_argument('--pinning-arm',choices=('optimized',),required=True)"),
        ('    changes = [', '''    changes = [
        ("parser.add_argument('--steps',type=int,default=3)", "parser.add_argument('--steps',type=int,default=3)\\n    parser.add_argument('--sft-numa-mode',choices=('unbound','gpu-local8','socket-split8'),required=True)\\n    parser.add_argument('--sft-numa-plan',type=Path,required=True)"),
        ("    command+=argv", "    command += ['--sft-numa-mode',args.sft_numa_mode,'--sft-numa-plan',str(args.sft_numa_plan)]\\n    command+=argv"),
        ("    # This survives SIGKILL/OOM", "    record.update(sft_numa_mode=args.sft_numa_mode,sft_numa_plan_sha256=hashlib.sha256(args.sft_numa_plan.read_bytes()).hexdigest(),incremental_baseline='accepted pinned repair')\\n    for name in ('sft_numa_entry_20261021.py','sft_numa_placement_20261021.py','numa_placement.py','run_sft_numa_case_20261021.py'):\\n        shutil.copyfile(PREP/name,snapshots/name)\\n    # This survives SIGKILL/OOM"),'''),
    ]
    for before, after in replacements:
        if text.count(before) != 1:
            raise ValueError('Exact SFT runner wrapper failed: '+before)
        text = text.replace(before, after)
    return text


def main():
    text = derive((PREP/'run_sft_trace_case_20261020.py').read_bytes())
    exec(compile(text, str(PREP/'run_sft_numa_case_20261021.generated.py'), 'exec'),
         {'__name__': '__main__', '__file__': __file__})


if __name__ == '__main__':
    main()
