"""Path-only dispatch of the frozen native SFT save/restore runner."""
import builtins
import os
from pathlib import Path
import sys

root = Path(os.environ['JIUGE_DT_ROOT'])
sys.path.insert(0, str(root))
import run_sft_pinning_case as original
stage = os.environ['JIUGE_DT_RESTORE_STAGE']
arm = os.environ['JIUGE_DT_ARM']
campaign = root/'CoreX-attention-fusion-20261028-03'
old = "str(PREP/'sft_pinning_entry.py'),'--evidence'"
def compiler(text, name, mode):
    assert text.count(old) == 1
    text = text.replace(old, "str(Path(os.environ['JIUGE_DT_CONTROL'])/'restore_entry.py'),'--evidence'")
    return builtins.compile(text, name, mode)
original.compile = compiler
sys.argv = [str(root/'run_sft_pinning_case.py'), '--case', f'corex-S2048-{stage}-{arm}-attention-rs01',
            '--mode', stage, '--pinning-arm', arm, '--pinning-campaign', str(Path(os.environ['JIUGE_DT_CONTROL'])/'pinning-backends'/f'{stage}-{arm}'),
            '--semantics-root', str(campaign/'semantics'/f'{stage}-{arm}'),
            '--steps', '3', '--seq-length', '2048']
sys.argv += ['--save-dir' if stage == 'checkpoint' else '--load-dir', str(campaign/'checkpoints')]
if stage == 'resume' and arm == 'optimized':
    sys.argv += ['--resume-audit-campaign', str(campaign/'numeric')]
original.main()
