"""Observe the unchanged native interval timer, with no tensor/profiler capture."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--jiuge-run-root', type=Path, required=True)
    parser.add_argument('--pinning-arm', choices=('native', 'control', 'optimized'), required=True)
    parser.add_argument('--pinning-campaign', type=Path, required=True)
    args, argv = parser.parse_known_args()
    rank = int(os.environ['RANK'])
    folder = args.jiuge_run_root / f'rank-{rank}'
    folder.mkdir(parents=True, exist_ok=False)
    evidence = Path(str(args.evidence).replace('{rank}', str(rank)))
    record = {'schema': 'jiuge7b-pinning-formal-rank-v1', 'rank': rank, 'arm': args.pinning_arm,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'argv': argv, 'status': 'initializing',
              'raw_interval_times': [], 'completed_steps': 0, 'skipped_steps': 0,
              'profiler_enabled': False, 'full_state_capture_enabled': False, 'checkpoint_io_enabled': False,
              'formal_performance_passed': False}
    root = PREP / 'megatron-native-5be9626'
    try:
        manifest = json.loads((PREP / 'source_manifest.json').read_text())
        if manifest['commit'] != '5be9626709af2722333bf54797c954c09edeada3':
            raise ValueError('Wrong frozen native commit')
        if sha(PREP / 'source_manifest.json') != 'a72ded960488d399ded2255ef55f357257e03532b8fa21a06d958170c5c98c2d':
            raise ValueError('Native source catalog differs')
        if any(sha(root / name) != digest for name, digest in manifest['files'].items()):
            raise ValueError('Frozen native source changed')
        if any(flag in argv for flag in ('--save', '--load', '--profile', '--record-memory-history')):
            raise ValueError('Audits/profiling/checkpoint IO forbidden in formal timing')
        record.update(native_files_checked=len(manifest['files']), source_manifest_sha256=sha(PREP / 'source_manifest.json'))
        os.environ.update(HF_HOME=str(PREP / 'huggingface-cache'), HF_MODULES_CACHE=str(PREP / 'huggingface-cache/modules'),
                          HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0', TOKENIZERS_PARALLELISM='false')
        sys.path.insert(0, str(root))
        import torch
        import megatron.training.argument_utils as argument_utils
        import megatron.training.training as training
        from megatron.core.timers import Timer
        import megatron.core.optimizer.cpu_offloading.hybrid_optimizer as module
        from jiuge7b_bridge import load_into_native
        from pinning_candidate import EXPECTED, SOURCE, load_class
        torch.backends.cuda.matmul.allow_tf32 = False
        if sha(SOURCE) != EXPECTED:
            raise ValueError('Original HDO source changed')
        hdo = module.HybridDeviceOptimizer
        if args.pinning_arm != 'native':
            name = 'candidate_hybrid_optimizer.py' if args.pinning_arm == 'optimized' else 'native_hybrid_optimizer.py'
            expected = '8c83ed05b1333b6775abea19bb2ff07f5966938764654ab133613fd52573ca88' if args.pinning_arm == 'optimized' else EXPECTED
            source = args.pinning_campaign / name
            if sha(source) != expected:
                raise ValueError('Independent HDO copy changed')
            hdo._get_sub_optimizer_param_groups = load_class(source, 'formal_hdo_copy')._get_sub_optimizer_param_groups
        group_method = hdo._get_sub_optimizer_param_groups
        record['dispatch_calls_before_training'] = 0
        def groups(*values, **keywords):
            record['dispatch_calls_before_training'] += 1
            return group_method(*values, **keywords)
        hdo._get_sub_optimizer_param_groups = groups
        config_from_args = argument_utils.gpt_config_from_args
        def configured(*values, **keywords):
            config = config_from_args(*values, **keywords)
            def load_models(models):
                if len(models) != 1:
                    raise ValueError('Exactly one native PP1 model chunk required')
                model = models[0]
                while hasattr(model, 'module'):
                    model = model.module
                record['weight_loading'] = load_into_native(model, PREP / 'checkpoints/9G7B_MHA', folder / 'weight-loading.json')
                return models
            config.pre_wrap_hooks = [*(config.pre_wrap_hooks or []), load_models]
            return config
        argument_utils.gpt_config_from_args = configured
        optimizers = []
        setup = training.setup_model_and_optimizer
        def observed_setup(*values, **keywords):
            result = setup(*values, **keywords)
            optimizers.append(result[1])
            return result
        training.setup_model_and_optimizer = observed_setup
        original_step = training.train_step
        def step(*values, **keywords):
            result = original_step(*values, **keywords)
            # No device reads, file IO, ranges or extra synchronization here.
            record['completed_steps'] += 1
            record['skipped_steps'] += int(result[1])
            return result
        training.train_step = step
        context = {'iteration': None, 'first': None}
        original_log = training.training_log
        def log(*values, **keywords):
            context['iteration'] = keywords.get('iteration', values[3] if len(values) > 3 else None)
            context['first'] = keywords.get('is_first_iteration', False)
            try:
                return original_log(*values, **keywords)
            finally:
                context['iteration'] = None
        training.training_log = log
        original_elapsed = Timer.elapsed
        def elapsed(timer, reset=True, barrier=False):
            result = original_elapsed(timer, reset=reset, barrier=barrier)
            if timer.name == 'interval-time' and context['iteration'] is not None and not context['first']:
                if not reset or not barrier or context['iteration'] % 10:
                    raise ValueError('Native timer interval semantics changed')
                record['raw_interval_times'].append({'end_iteration': context['iteration'], 'steps': 10, 'seconds': result,
                                                     'native_reset': reset, 'native_barrier': barrier})
            return result
        Timer.elapsed = elapsed
        sys.argv = [str(root / 'pretrain_gpt.py'), *argv]
        runpy.run_path(str(root / 'pretrain_gpt.py'), run_name='__main__')
        if record['completed_steps'] != 120 or record['skipped_steps']:
            raise ValueError('120 actual unskipped native training steps required')
        if [row['end_iteration'] for row in record['raw_interval_times']] != list(range(10, 121, 10)):
            raise ValueError('Incomplete native timer windows')
        if len(optimizers) != 1 or record['dispatch_calls_before_training'] < 1:
            raise ValueError('Actual optimizer dispatch missing')
        from optimizer_evidence import save_optimizer_evidence
        from pinning_resume_audit import optimizer_leaves
        record['actual_optimizer_state'] = save_optimizer_evidence(optimizers[0], folder / 'optimizer-state.json')
        leaves = [opt for opt in optimizer_leaves(optimizers[0]) if isinstance(opt, hdo)]
        if len(leaves) != 1:
            raise ValueError('One actual HDO required')
        masters = list(leaves[0].gpu_params_map_cpu_copy.values())
        record['actual_final_cpu_masters'] = {'groups': len(masters), 'elements': sum(p.numel() for p in masters),
                                              'pinned_groups': sum(p.is_pinned() for p in masters)}
        if len(masters) != 199 or sum(p.numel() for p in masters) != 3834518016:
            raise ValueError('Actual final optimizer coverage differs')
        if args.pinning_arm == 'optimized' and not all(p.is_pinned() for p in masters):
            raise ValueError('Pinned candidate dispatch did not retain pinned master buffers')
        record.update(status='completed', gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as exc:
        record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        record.update(finished_utc=datetime.now(timezone.utc).isoformat(), launcher_sha256=sha(Path(__file__)))
        with evidence.open('x') as stream:
            json.dump(record, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
