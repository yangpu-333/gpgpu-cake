"""Exercise the actual disabled native initializer with fused=False/True.

Each invocation uses the unchanged five-update/full-state/native-resume fixture
on both NUMA nodes and performs the actual opposite-to-local thread switch.
Run twice with different JIUGE_PHASE_FIXTURE_ROOT and JIUGE_FIXTURE_FUSED.
"""
import os
from check_sft_phased_moments_20261023 import generated as previous


def generated():
    text = previous()
    fused = os.environ['JIUGE_FIXTURE_FUSED']
    if fused not in ('0', '1'):
        raise ValueError('Explicit native fused setting required')
    changes = (
        ('from sft_local_moments_v3_20261021 import install,node_for_current_cpus,relocate_moment',
         'from sft_local_moments_v4_20261023 import install,node_for_current_cpus,relocate_moment'),
        ("PREP/'sft_local_moments_v3_20261021.py'", "PREP/'sft_local_moments_v4_20261023.py'"),
        ('lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1)',
         'lr=1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused='+str(fused == '1')+')'),
        ('    uninstall = install(torch,events)',
         '    import inspect,torch.optim.adamw as native_adamw\n'
         '    native_initializer=inspect.unwrap(native_adamw.AdamW._init_group)\n'
         '    native_adamw.AdamW._init_group=torch._dynamo.disable(native_initializer)\n'
         "    assert native_adamw.AdamW._init_group.__globals__['__name__']=='torch._dynamo.eval_frame'\n"
         "    assert '_device_dtype_check_for_fused' in native_initializer.__globals__\n"
         '    disabled_before=native_adamw.AdamW._init_group\n'
         '    uninstall = install(torch,events)'),
        ('        uninstall()',
         '        uninstall()\n'
         '        assert native_adamw.AdamW._init_group is disabled_before'),
        ("'actual_thread_phase_switch_verified':True,'passed':True",
         "'actual_thread_phase_switch_verified':True,'native_disabled_initializer_exercised':True,'fused':"+str(fused == '1')+",'passed':True"),
        ("'node':x['node'],'actual_thread_phase_switch_verified'",
         "'node':x['node'],'fused':x['fused'],'native_disabled_initializer_exercised':x['native_disabled_initializer_exercised'],'actual_thread_phase_switch_verified'"),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Exact wrapped-initializer CPU fixture required: '+before)
        text = text.replace(before, after)
    return text


if __name__ == '__main__':
    exec(compile(generated(), '<native-disabled-initializer-state-resume>', 'exec'),
         {'__name__': '__main__', '__file__': __file__})
