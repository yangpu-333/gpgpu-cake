"""Run original pretrain_gpt with explicit BF16 diagnostics and opt-in kernels."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT.parent/'megatron_training_entry'
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--megatron-root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--entry-evidence',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--audit-dir',type=Path)
    parser.add_argument('--profile-dir',type=Path)
    opts,argv = parser.parse_known_args()
    root = opts.megatron_root.resolve()
    if opts.entry_evidence.exists():
        parser.error('immutable output already exists')
    contract = json.loads(opts.contract.read_text())
    report = {'status':'initializing','started_utc':datetime.now(timezone.utc).isoformat(),
        'megatron_commit':(root/'.git/HEAD').read_text().strip(),'formal_entry':str(root/'pretrain_gpt.py'),
        'training_argv':argv,'iluvatar_kernels':opts.candidate is not None,'models':[],
        'contract_sha256':sha(opts.contract),'contract_path':str(opts.contract.resolve())}
    handles,probes,models = [],[],[]
    profiler = None
    original_step = None
    training = None
    try:
        if report['megatron_commit'] != COMMIT or contract['megatron_commit'] != COMMIT:
            raise RuntimeError('fixed native commit required')
        manifest = json.loads((FORMAL/'source_manifest.json').read_text())
        observed = {name:sha(root/name) for name in manifest['files']}
        if observed != manifest['files']:
            raise RuntimeError('native source differs from exact 600-file manifest')
        report['native_sources_sha256'] = observed
        report['integration_sha256'] = {name:sha(ROOT/name) for name in
            ('launch_training.py','kernel_adapter.py','run_case.py')}
        report['integration_sha256'].update({'formal/'+name:sha(FORMAL/name) for name in
            ('runtime_compat.py','source_manifest.json')})
        sys.path.insert(0,str(FORMAL))
        from runtime_compat import prepare_local_runtime
        report['runtime_compatibility'] = prepare_local_runtime(root)
        # The sibling FP32 suite also has kernel_adapter.py. Resolve this
        # suite's modules first after importing its frozen compatibility code.
        sys.path.insert(0,str(ROOT))
        import torch
        from megatron.training import argument_utils
        import megatron.training.training as training
        original_config = argument_utils.gpt_config_from_args
        def configured(*args,**kwargs):
            config = original_config(*args,**kwargs)
            def install(chunk_list):
                for index,model in enumerate(chunk_list):
                    models.append(model)
                    report['models'].append({'class':type(model).__module__+'.'+type(model).__name__,
                        'parameters':sum(p.numel() for p in model.parameters()),
                        'parameter_names':list(dict(model.named_parameters())),
                        'state_dict_keys':list(model.state_dict()),'layers':len(model.decoder.layers),
                        'hidden_size':model.config.hidden_size,'vocab_size':model.vocab_size,
                        'bf16':model.config.bf16,'fp16':model.config.fp16})
                    if opts.candidate is not None:
                        from kernel_adapter import install_kernels
                        handles.append(install_kernels(model,opts.candidate,contract,root))
                    if opts.profile_dir is not None:
                        from profile_hooks import install_profile_hooks
                        probes.append(install_profile_hooks(model))
                    if opts.audit_dir is not None:
                        opts.audit_dir.mkdir(parents=True,exist_ok=True)
                        def cpu(value):
                            return value.detach().cpu().clone() if torch.is_tensor(value) else value
                        torch.save({name:cpu(value) for name,value in model.state_dict().items()},
                                   opts.audit_dir/('model%d-initial.pt'%index))
                        count=[0]
                        def forward_capture(module,inputs,keywords,output,model_index=index,counter=count):
                            counter[0]+=1
                            torch.save({'output':cpu(output),
                                'positional_inputs':{str(i):cpu(v) if torch.is_tensor(v) else None for i,v in enumerate(inputs)},
                                'keyword_inputs':{name:cpu(v) for name,v in keywords.items() if torch.is_tensor(v)}},
                                opts.audit_dir/('model%d-forward-%03d.pt'%(model_index,counter[0])))
                        model.register_forward_hook(forward_capture,with_kwargs=True)
                return chunk_list
            config.pre_wrap_hooks = list(config.pre_wrap_hooks or [])+[install]
            return config
        argument_utils.gpt_config_from_args = configured
        if opts.audit_dir is not None or opts.profile_dir is not None:
            original_step = training.train_step
            signature = inspect.signature(original_step)
            count=[0]
            def states(optimizer):
                named = {id(value):(name,value) for model in models for name,value in model.named_parameters()}
                master = {}
                children = getattr(optimizer,'chained_optimizers',[optimizer])
                for child in children:
                    low_groups = getattr(child,'float16_groups',[])
                    main_groups = getattr(child,'fp32_from_float16_groups',[])
                    if len(low_groups)!=len(main_groups):
                        raise RuntimeError('native master group count mismatch')
                    for low,high in zip(low_groups,main_groups):
                        if len(low)!=len(high):
                            raise RuntimeError('native master parameter count mismatch')
                        for param,main in zip(low,high):
                            name = named[id(param)][0]
                            if name in master:
                                raise RuntimeError('duplicate native master parameter')
                            master[name] = main.detach().cpu().clone()
                    for group in getattr(child,'fp32_from_fp32_groups',[]):
                        for param in group:
                            master[named[id(param)][0]] = param.detach().cpu().clone()
                if set(master) != {name for name,_ in named.values()}:
                    raise RuntimeError('native master mapping must cover every model parameter')
                return {'model_parameters':{name:p.detach().cpu().clone() for name,p in named.values()},
                        'master_parameters':master}
            def measured_step(*args,**kwargs):
                bound = signature.bind(*args,**kwargs)
                count[0]+=1
                if opts.audit_dir is not None and count[0]==1:
                    torch.save(states(bound.arguments['optimizer']),opts.audit_dir/'actual-start.pt')
                result = original_step(*args,**kwargs)
                if opts.audit_dir is not None:
                    torch.save(states(bound.arguments['optimizer'])['master_parameters'],
                               opts.audit_dir/('master-after-%03d.pt'%count[0]))
                if profiler is not None:
                    profiler.step()
                return result
            training.train_step = measured_step
        if opts.profile_dir is not None:
            opts.profile_dir.mkdir(parents=True,exist_ok=True)
            supported = torch.profiler.supported_activities()
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.profiler.ProfilerActivity.CUDA in supported:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            report['profiler_activities'] = [str(value) for value in activities]
            def trace_ready(active):
                active.export_chrome_trace(str(opts.profile_dir/'trace.json'))
                rows=[]
                for event in active.key_averages(group_by_input_shape=True):
                    rows.append({'name':event.key,'count':event.count,'input_shapes':event.input_shapes,
                        'self_cpu_us':event.self_cpu_time_total,'cpu_total_us':event.cpu_time_total,
                        'self_device_us':getattr(event,'self_device_time_total',getattr(event,'self_cuda_time_total',0)),
                        'device_total_us':getattr(event,'device_time_total',getattr(event,'cuda_time_total',0))})
                (opts.profile_dir/'events.json').write_text(json.dumps(rows,indent=2)+'\n')
            profiler = torch.profiler.profile(activities=activities,
                schedule=torch.profiler.schedule(wait=2,warmup=1,active=3,repeat=1),
                record_shapes=True,on_trace_ready=trace_ready)
            profiler.start()
        sys.argv = [str(root/'pretrain_gpt.py'),*argv]
        runpy.run_path(str(root/'pretrain_gpt.py'),run_name='__main__')
        report['status']='completed'
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        if profiler is not None:
            profiler.stop()
        if original_step is not None:
            training.train_step=original_step
        report['kernel_adapters']=[handle.snapshot() for handle in handles]
        report['profile_hooks']=[handle.snapshot() for handle in probes]
        report['launcher_sha256']=sha(__file__)
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        opts.entry_evidence.parent.mkdir(parents=True,exist_ok=True)
        opts.entry_evidence.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
