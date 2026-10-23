"""Frozen CPU state/resume fixture plus real opposite-to-local thread binding."""
import json
import os
from pathlib import Path
import sys
from check_sft_local_moments_v3_20261021 import generated as previous

BASE = Path(__file__).resolve().parent


def cpu_plan(path, node):
    allowed=sorted(os.sched_getaffinity(0))
    topology={}
    cores={}
    for cpu in allowed:
        memberships=[]
        for folder in Path('/sys/devices/system/node').glob('node[0-9]*'):
            from sft_numa_placement_20261021 import cpulist
            if cpu in cpulist((folder/'cpulist').read_text()):
                memberships.append(int(folder.name[4:]))
        if len(memberships)!=1:raise ValueError('Known actual CPU NUMA membership required')
        t=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        item={'node':memberships[0],'package':(t/'physical_package_id').read_text().strip(),
              'core':(t/'core_id').read_text().strip()}
        topology[str(cpu)]=item
        cores.setdefault(item['node'],{}).setdefault((item['package'],item['core']),cpu)
    selected=sorted(cores[node].values())
    if len(selected)<16:raise ValueError('CPU fixture requires the actual16 physical node cores')
    plan={'CPU_fixture_only':True,'GPU_devices_queried':False,
          'topology':{'allowed_cpus':allowed,'cpu_topology':topology,
                      'gpus':[{'node':node,'uuid':'CPU-fixture-no-GPU-'+str(r)} for r in (0,1)]},
          'policies':{'unbound':[allowed[:],allowed[:]],'gpu-local8':[selected[:8],selected[8:16]]}}
    with Path(path).open('x') as stream:json.dump(plan,stream,indent=2)
    return Path(path)


def generated():
    text=previous()
    changes=(
        ("ROOT = PREP/'sft-moment-direct-zero-preparation-20261021-01/cpu-regression'",
         "ROOT = Path(os.environ['JIUGE_PHASE_FIXTURE_ROOT'])"),
        ('    os.sched_setaffinity(0,{cpu})',
         "    from check_sft_phased_moments_20261023 import cpu_plan\n    from sft_phased_placement_20261023 import prepare,activate,check_node_values\n    plan=cpu_plan(ROOT/f'plan-node-{node}.json',node)\n    initial_binding=prepare(plan,'gpu-local8',0)\n    assert initial_binding['selected_cpu_nodes']!=[node]"),
        ("OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'",
         "OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8'"),
        ('assert node_for_current_cpus()==node and not torch.cuda.is_initialized()',
         'assert node_for_current_cpus()!=node and not torch.cuda.is_initialized()'),
        ('    uninstall = install(torch,events)',
         "    final_binding=activate(plan,'gpu-local8',0,folder)\n    assert final_binding['selected_cpu_nodes']==[node]\n    assert len(final_binding['allowed_after'])==8\n    assert all(v==final_binding['allowed_after'] for v in final_binding['thread_affinity_after'].values())\n    assert node_for_current_cpus()==node\n    uninstall = install(torch,events)"),
        ("    value = {'passed':True",
         "    value = {'initial_binding':initial_binding,'update_binding':final_binding,'actual_thread_phase_switch_verified':True,'passed':True"),
        ("assert socket.gethostname()=='sg-huids25-260917-94135-default0-0'",
         "assert socket.gethostname()==os.environ['JIUGE_PHASE_FIXTURE_HOST']"),
        ("'five_updates_full_state_exact':x['five_updates_full_state_exact'],",
         "'actual_thread_phase_switch_verified':x['actual_thread_phase_switch_verified'],'five_updates_full_state_exact':x['five_updates_full_state_exact'],"),
    )
    for before,after in changes:
        if text.count(before)!=1:raise ValueError('Exact phased CPU fixture adaptation: '+before)
        text=text.replace(before,after)
    return text


if __name__=='__main__':
    exec(compile(generated(),'<unchanged-state-resume-with-thread-phases>','exec'),
         {'__name__':'__main__','__file__':__file__})
