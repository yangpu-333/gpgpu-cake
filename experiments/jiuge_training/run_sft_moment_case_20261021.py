"""Add a storage-only arm to the frozen35-update SFT capture runner."""
from pathlib import Path
from run_sft_numa_case_20261021 import derive as numa_derive

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def derive(source):
    text=numa_derive(source)
    before="str(PREP/'sft_numa_entry_20261021.py')"
    if text.count(before)!=1:raise ValueError('Expected one generated entry path')
    text=text.replace(before,"str(PREP/'sft_moment_entry_20261021.py')")
    extra=[
        ("parser.add_argument('--steps',type=int,default=3)","parser.add_argument('--steps',type=int,default=3)\n    parser.add_argument('--sft-moment-arm',choices=('baseline','affinity-control','local-moments'),required=True)"),
        ("    command+=argv","    command += ['--sft-moment-arm',args.sft_moment_arm]\n    command+=argv"),
        ("    # This survives SIGKILL/OOM","    record.update(sft_moment_arm=args.sft_moment_arm)\n    for name in ('sft_local_moments_20261021.py','sft_moment_entry_20261021.py','run_sft_moment_case_20261021.py'):\n        shutil.copyfile(PREP/name,snapshots/name)\n    # This survives SIGKILL/OOM")]
    before='    changes = ['
    if text.count(before)!=1:raise ValueError('Expected one outer change list')
    text=text.replace(before,before+'\n'+''.join('        '+repr(pair)+',\n' for pair in extra))
    return text


def main():
    text=derive((PREP/'run_sft_trace_case_20261020.py').read_bytes())
    exec(compile(text,str(PREP/'run_sft_moment_case_20261021.generated.py'),'exec'),
         {'__name__':'__main__','__file__':__file__})


if __name__=='__main__':main()
