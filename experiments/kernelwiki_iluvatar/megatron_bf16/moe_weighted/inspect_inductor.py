"""Untimed inspect of the retained original Inductor weighted-backward kernel config."""
import hashlib,importlib.util,json
from pathlib import Path
import torch
path=Path('/tmp/torchinductor_root/da/cda4ewgdc6jebzdqqp2r465jonsesjsrpfhbjgt3p2o2etmm2gji.py')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='1df526ea803e1130c46352133c855d84377d1d07b88d3800eb848a1404c72bb0'
spec=importlib.util.spec_from_file_location('original_weighted_bwd_codegen_config',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
k=m.triton_red_fused_mul_silu_sum_1
r={'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'class':str(type(k)),
'configs':[{'kwargs':z.kwargs,'num_warps':z.num_warps,'num_stages':z.num_stages} for z in k.configs],
'launchers':len(k.launchers),'schema':'original-inductor-weighted-reduction-config-v1'}
print(json.dumps(r));out=Path(__file__).parent/'original-inductor-config.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n')
