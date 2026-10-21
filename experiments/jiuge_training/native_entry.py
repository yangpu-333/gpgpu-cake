"""Run the pinned native Megatron entry with immutable process evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
COMMIT = '5be9626709af2722333bf54797c954c09edeada3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--native-script', choices=('pretrain_gpt.py', 'tools/preprocess_data.py'), default='pretrain_gpt.py')
    parser.add_argument('--jiuge-run-root', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--capture-windows', action='store_true')
    parser.add_argument('--capture-inputs', action='store_true')
    args, remainder = parser.parse_known_args()
    args_capture_inputs = args.capture_inputs
    if args.capture_inputs and args.capture_windows:
        raise ValueError('Input copies belong to a separate admission run')
    rank = int(os.environ.get('RANK', '0'))
    args.evidence = Path(str(args.evidence).replace('{rank}', str(rank)))
    if args.evidence.exists():
        raise ValueError('Immutable evidence path already exists')
    root = PREP / 'megatron-native-5be9626'
    record = {'started_utc': datetime.now(timezone.utc).isoformat(),
              'entry': str(root / args.native_script), 'argv': remainder, 'status': 'initializing'}
    capture = None
    created_optimizers = []
    completed_steps = []
    input_rows = []
    try:
        manifest = json.loads((PREP / 'source_manifest.json').read_text())
        if manifest['commit'] != COMMIT or (root / '.git/HEAD').read_text().strip() != COMMIT:
            raise ValueError('Pinned native commit required')
        mismatches = [name for name, value in manifest['files'].items() if sha(root / name) != value]
        if mismatches:
            raise ValueError('Pinned source mismatch: ' + str(mismatches))
        record.update(commit=COMMIT, native_files_checked=len(manifest['files']),
                      source_manifest_sha256=sha(PREP / 'source_manifest.json'))
        os.environ.update(HF_HOME=str(PREP / 'huggingface-cache'),
                          HF_MODULES_CACHE=str(PREP / 'huggingface-cache/modules'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
                          TOKENIZERS_PARALLELISM='false')
        sys.path.insert(0, str(root))
        if args.jiuge_run_root:
            import torch
            import megatron.training.argument_utils as argument_utils
            import megatron.training.training as training
            from jiuge7b_bridge import load_into_native, compare_forward_reference
            torch.backends.cuda.matmul.allow_tf32 = False
            rank_root = args.jiuge_run_root / f'rank-{rank}'
            rank_root.mkdir(parents=True,exist_ok=True)
            original_config = argument_utils.gpt_config_from_args
            def configured(*values,**keywords):
                config = original_config(*values,**keywords)
                def load_models(models):
                    if len(models) != 1:
                        raise ValueError('PP1 one native chunk required')
                    model = models[0]
                    while hasattr(model,'module'):
                        model = model.module
                    record['weight_loading'] = load_into_native(model,PREP/'checkpoints/9G7B_MHA',rank_root/'weight-loading.json')
                    if args.reference_dir:
                        record['forward_equivalence'] = compare_forward_reference(model,args.reference_dir,rank_root/'forward-equivalence.json')
                    return models
                config.pre_wrap_hooks = [*(config.pre_wrap_hooks or []),load_models]
                return config
            argument_utils.gpt_config_from_args = configured
            original_step = training.train_step
            count = 0
            if args.capture_windows:
                from trace_hooks import CaptureWindows
                capture = CaptureWindows(torch,args.jiuge_run_root,rank,args.jiuge_run_root.name)
            original_setup = training.setup_model_and_optimizer
            def setup_with_ranges(*values,**keywords):
                result = original_setup(*values,**keywords)
                optimizer = result[1]
                if optimizer is not None:
                    created_optimizers.append(optimizer)
                    original_optimizer_step = optimizer.step
                    def optimizer_step(*args,**kwargs):
                        with torch.profiler.record_function('jiuge/optimizer_update'):
                            return original_optimizer_step(*args,**kwargs)
                    optimizer.step = optimizer_step
                return result
            training.setup_model_and_optimizer = setup_with_ranges
            original_get_schedule = training.get_forward_backward_func
            def get_schedule_with_ranges(*values,**keywords):
                original = original_get_schedule(*values,**keywords)
                def schedule(*args,**kwargs):
                    if args_capture_inputs:
                        from input_evidence import install_input_observer
                        callback = kwargs.get('forward_step_func') or (args[0] if args else None)
                        install_input_observer(torch,callback,rank_root,input_rows)
                    with torch.profiler.record_function('jiuge/forward_backward'):
                        return original(*args,**kwargs)
                return schedule
            training.get_forward_backward_func = get_schedule_with_ranges
            def observed_step(*values,**keywords):
                nonlocal count
                count += 1
                print(json.dumps({'native_step_start':count,'rank':rank}),flush=True)
                if capture:
                    capture.before(count)
                with torch.profiler.record_function(f'jiuge/train_step/{count}'):
                    result = original_step(*values,**keywords)
                if capture:
                    capture.after(count)
                print(json.dumps({'native_step_complete':count,'rank':rank}),flush=True)
                completed_steps.append({'step':count,'skipped':int(result[1])})
                return result
            training.train_step = observed_step
        # Upstream help contains a literal percent sign invalid for argparse's
        # interpolation. This fallback affects help rendering only, not parsing.
        original_expand = argparse.HelpFormatter._expand_help
        def literal_help_fallback(formatter, action):
            try:
                return original_expand(formatter, action)
            except (ValueError, TypeError):
                return formatter._get_help_string(action)
        argparse.HelpFormatter._expand_help = literal_help_fallback
        record['help_only_formatting_compatibility'] = True
        sys.argv = [str(root / args.native_script), *remainder]
        runpy.run_path(str(root / args.native_script), run_name='__main__')
        if created_optimizers:
            from optimizer_evidence import save_optimizer_evidence
            record['actual_optimizer_state'] = save_optimizer_evidence(created_optimizers[-1],rank_root/'optimizer-state.json')
            record['completed_native_steps'] = completed_steps
            if args_capture_inputs:
                record['actual_input_batches'] = input_rows
            record['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            record['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        record['status'] = 'completed'
    except SystemExit as exc:
        record.update(status='completed' if exc.code in (None, 0) else 'failed', exit_code=exc.code)
        raise
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if capture:
            capture.close()
            record['captures'] = capture.receipts
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), launcher_sha256=sha(__file__))
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.evidence.open('x') as stream:
            json.dump(record, stream, indent=2)


if __name__ == '__main__':
    main()
