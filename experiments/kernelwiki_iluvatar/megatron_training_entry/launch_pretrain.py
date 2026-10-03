"""Launch pinned pretrain_gpt.py with an explicit opt-in kernel integration.

Megatron keeps ownership of argument parsing, dataset construction, distributed
wrapping, forward/backward scheduling, optimizer, checkpoints and train loop.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
EXPECTED_COMMIT = '5be9626709af2722333bf54797c954c09edeada3'


def main():
    parser = argparse.ArgumentParser(add_help=False, description=__doc__)
    parser.add_argument('--megatron-root', type=Path, required=True)
    parser.add_argument('--corex42-compat', action='store_true')
    parser.add_argument('--iluvatar-kernels', action='store_true')
    parser.add_argument('--entry-evidence', type=Path, required=True)
    parser.add_argument('--entry-tensor-audit-dir', type=Path)
    opts, training_args = parser.parse_known_args()
    root = opts.megatron_root.resolve()
    if opts.entry_evidence.exists():
        parser.error('refusing to overwrite launch evidence')
    # The authorized remote checkout is detached; older vendor Git builds do
    # not honor process-only safe.directory. Read the fixed HEAD directly.
    commit = (root/'.git/HEAD').read_text(encoding='utf-8').strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError('unexpected Megatron commit')
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'megatron_commit': commit,
              'formal_entry': str(root/'pretrain_gpt.py'), 'training_argv': training_args,
              'iluvatar_kernels': opts.iluvatar_kernels, 'status':'initializing'}
    handles = []
    model_records = []
    try:
        manifest_path = ROOT/'source_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['commit'] != commit:
            raise RuntimeError('source manifest commit differs')
        observed = {name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in manifest['files']}
        changed = [name for name in observed if observed[name] != manifest['files'][name]]
        if changed:
            raise RuntimeError('native source integrity mismatch: '+repr(changed))
        report['native_sources_sha256'] = observed
        report['integration_sha256'] = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                       for name in ('launch_pretrain.py','runtime_compat.py',
                                                    'kernel_adapter.py','run_case.py','contract.json',
                                                    'source_manifest.json')}
        if opts.corex42_compat:
            if '--transformer-impl' not in training_args or training_args[training_args.index('--transformer-impl')+1] != 'local':
                raise ValueError('CoreX compatibility requires explicit --transformer-impl local')
            from runtime_compat import prepare_local_runtime
            report['runtime_compatibility'] = prepare_local_runtime(root)
        else:
            sys.path.insert(0,str(root))
        from megatron.training import argument_utils
        original_config = argument_utils.gpt_config_from_args
        def config_with_kernels(*args, **kwargs):
            config = original_config(*args, **kwargs)
            def install_before_wrap(models):
                import torch
                for index, model in enumerate(models):
                    model_records.append({'class':type(model).__module__+'.'+type(model).__name__,
                                          'parameters':sum(p.numel() for p in model.parameters()),
                                          'state_dict_keys':list(model.state_dict()),
                                          'layers':len(model.decoder.layers),
                                          'hidden_size':model.config.hidden_size,
                                          'vocab_size':model.vocab_size})
                    if opts.iluvatar_kernels:
                        from kernel_adapter import install_kernels
                        handles.append(install_kernels(model, enabled=True, megatron_root=root))
                    if opts.entry_tensor_audit_dir is not None:
                        target = opts.entry_tensor_audit_dir
                        target.mkdir(parents=True,exist_ok=True)
                        initial = target/('model%d-initial.pt'%index)
                        if initial.exists():
                            raise FileExistsError(initial)
                        torch.save({name:(value.detach().cpu().clone() if torch.is_tensor(value) else value)
                                    for name,value in model.state_dict().items()},initial)
                        forward_count=[0]
                        def capture_forward(module, inputs, keyword_inputs, output, model_index=index, counter=forward_count):
                            counter[0]+=1
                            destination=target/('model%d-forward-%03d.pt'%(model_index,counter[0]))
                            if destination.exists():
                                raise FileExistsError(destination)
                            def cpu(value):
                                return value.detach().cpu().clone() if torch.is_tensor(value) else None
                            torch.save({'output':cpu(output),
                                        'positional_inputs':{str(i):cpu(v) for i,v in enumerate(inputs)},
                                        'keyword_inputs':{key:cpu(value) for key,value in keyword_inputs.items()
                                                          if torch.is_tensor(value)}},destination)
                        model.register_forward_hook(capture_forward,with_kwargs=True)
                return models
            config.pre_wrap_hooks = list(config.pre_wrap_hooks or []) + [install_before_wrap]
            return config
        argument_utils.gpt_config_from_args = config_with_kernels
        sys.argv = [str(root/'pretrain_gpt.py'), *training_args]
        runpy.run_path(str(root/'pretrain_gpt.py'), run_name='__main__')
        report['status'] = 'completed'
    except SystemExit as error:
        report['status'] = 'completed' if error.code in (None,0) else 'failed'
        report['exit_code'] = error.code
        raise
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        report['kernel_adapters'] = [handle.snapshot() for handle in handles]
        report['models'] = model_records
        report['launcher_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        opts.entry_evidence.parent.mkdir(parents=True,exist_ok=True)
        opts.entry_evidence.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
