"""Explicit process-only gradnorm hook around unchanged BF16 training launcher."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    p.add_argument('--bf16-root',type=Path,required=True)
    p.add_argument('--gradnorm-candidate',type=Path)
    p.add_argument('--gradnorm-receipt',type=Path,required=True)
    a,rest=p.parse_known_args()
    assert not a.gradnorm_receipt.exists()
    root=a.bf16_root.resolve()
    sys.path.insert(0,str(root.parent/'megatron_training_entry'))
    import runtime_compat
    original_prepare=runtime_compat.prepare_local_runtime
    receipt={'enabled':a.gradnorm_candidate is not None,'calls':0,
             'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'candidate_sha256':hashlib.sha256(a.gradnorm_candidate.read_bytes()).hexdigest() if a.gradnorm_candidate else None,
             'scope':'Process-only local multi-tensor L2 hook; original all-reduces, optimizer, CE0008 and training timer unchanged.'}
    def prepared(native_root):
        result=original_prepare(native_root)
        if a.gradnorm_candidate:
            import megatron.core.utils as utils
            import megatron.core.optimizer.clip_grads as clip
            native=utils.local_multi_tensor_l2_norm
            spec=importlib.util.spec_from_file_location('candidate_gradnorm',a.gradnorm_candidate)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            def call(*args,**kwargs):
                receipt['calls']+=1
                return module.local_multi_tensor_l2_norm(*args,native=native,**kwargs)
            assert clip.l2_norm_impl is native
            utils.local_multi_tensor_l2_norm=call
            clip.l2_norm_impl=call
        return result
    runtime_compat.prepare_local_runtime=prepared
    sys.argv=[str(root/'launch_training.py')]+rest
    try:
        runpy.run_path(str(root/'launch_training.py'),run_name='__main__')
    finally:
        a.gradnorm_receipt.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
