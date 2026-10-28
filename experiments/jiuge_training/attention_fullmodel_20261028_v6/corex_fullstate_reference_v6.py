"""Bind a fresh strict baseline to accepted native raw reference values.

The original chunk capture, finite check and CPU-float64 comparison are called
unchanged. New native gets the additional reference comparison as well.
"""
import copy,json
from pathlib import Path
def install(module,campaign,rank):
 original=module.FullAudit;campaign=Path(campaign)
 class ReferenceBoundAudit(original):
  def __init__(self,root,arm,actual_rank):
   super().__init__(root,arm,actual_rank)
   if arm=='native':
    old=json.loads((campaign/'reference/native'/f'rank-{actual_rank}'/'all-state-manifest.json').read_text())
    assert old['step']==3 and old['passed_so_far'] and len(old['tensors'])==3585
    rows=copy.deepcopy(old['tensors'])
    for row in rows:
     for chunk in row['chunks']:chunk['file']='reference/'+chunk['file']
    self.reference={x['field']:x for x in rows}
  def tensor(self,field,tensor):
   if field not in self.reference:raise ValueError('Complete accepted reference field required: '+field)
   return super().tensor(field,tensor)
 module.FullAudit=ReferenceBoundAudit
 return lambda:setattr(module,'FullAudit',original)
