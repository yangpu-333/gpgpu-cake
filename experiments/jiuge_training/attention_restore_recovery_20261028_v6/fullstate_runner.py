import builtins,os,sys
from pathlib import Path
R=Path(os.environ['JIUGE_DT_ROOT']);C=Path(os.environ['JIUGE_DT_CONTROL']);CAMP=Path(os.environ['JIUGE_ATTENTION_CAMPAIGN']);sys.path.insert(0,str(R));import run_sft_pinning_case as original
arm=os.environ['JIUGE_DT_ARM'];needle="str(PREP/'sft_pinning_entry.py'),'--evidence'"
def compiler(text,name,mode):
 assert text.count(needle)==1;text=text.replace(needle,"str(Path(os.environ['JIUGE_DT_CONTROL'])/'fullstate_entry.py'),'--evidence'")
 return builtins.compile(text,name,mode)
original.compile=compiler
sys.argv=[str(R/'run_sft_pinning_case.py'),'--case',f'corex-S2048-attention-audit-{arm}-af03','--mode','audit','--pinning-arm',arm,'--pinning-campaign',str(CAMP/'numeric'),'--semantics-root',str(CAMP/'semantics'/arm),'--steps','3','--seq-length','2048']
original.main()
