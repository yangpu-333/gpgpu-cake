"""One native softmax forward/backward per shape; no profiling or optimization.

Default fusion and explicitly requested native fallback are separate invocations.
Only instance-level route observers are added; pinned sources and native math
are never modified. CUDA-extension API coverage is not a GPU timeline claim.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


PIN = '5be9626709af2722333bf54797c954c09edeada3'
EXTENSIONS = ('scaled_masked_softmax_cuda', 'scaled_upper_triang_masked_softmax_cuda',
              'scaled_softmax_cuda')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extension_state():
    results = {}
    for name in EXTENSIONS:
        module = sys.modules.get(name)
        try:
            spec = importlib.util.find_spec(name)
            importable = spec is not None
        except (ImportError, ValueError):
            importable = module is not None
        results[name] = {'importable': importable, 'loaded': module is not None}
        if module is not None and getattr(module, '__file__', None):
            path = Path(module.__file__).resolve()
            results[name].update(path=str(path), sha256=sha(path))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('default-fusion', 'native-fallback'),
                        default='default-fusion')
    parser.add_argument('--sequences', type=int, nargs='+', default=(128, 256, 512))
    parser.add_argument('--megatron-root', type=Path,
                        default=Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'))
    parser.add_argument('--runtime-root', type=Path,
                        default=Path('/tmp/kda-bf16-continuation/megatron_training_entry'))
    args = parser.parse_args()
    assert not args.output.exists(), 'immutable backend receipt already exists'
    assert all(0 < sequence <= 4096 for sequence in args.sequences)
    manifest_path = args.runtime_root / 'source_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['commit'] == PIN
    sources = {name: sha(args.megatron_root/name) for name in
               ('megatron/core/fusions/fused_softmax.py',
                'megatron/core/transformer/utils.py',
                'megatron/core/transformer/dot_product_attention.py')}
    assert all(value == manifest['files'][name] for name, value in sources.items())
    receipt = {'schema': 'native-attention-softmax-backend-probe-v1',
               'requested_backend': args.backend, 'megatron_commit': PIN,
               'source_manifest_sha256': sha(manifest_path), 'native_source_sha256': sources,
               'controller_sha256': sha(Path(__file__)),
               'started_utc': datetime.now(timezone.utc).isoformat(), 'cases': [],
               'note': 'One forward and one backward per shape, no performance measurement. '
                       'CUDA-extension API coverage does not prove a GPU kernel timeline. '
                       'Fallback is never silently substituted for requested default fusion.'}
    try:
        with Path('/tmp/kda-biv150-gpu0.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            sys.path.insert(0, str(args.runtime_root))
            from runtime_compat import prepare_local_runtime
            receipt['runtime_compatibility'] = prepare_local_runtime(args.megatron_root)
            import torch
            torch.cuda.set_device(0)
            receipt.update(torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
                           extensions_before=extension_state())
            assert receipt['gpu'] in ('BI-V150', 'Iluvatar BI-V150')
            from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
            from megatron.core.transformer.enums import AttnMaskType
            from megatron.core.transformer.utils import attention_mask_func
            torch.manual_seed(31102)
            for sequence in args.sequences:
                row = {'sequence': sequence, 'shape': [2, 8, sequence, sequence],
                       'mask_shape': [2, 1, sequence, sequence], 'mask_true_means': 'masked',
                       'input_dtype': 'torch.bfloat16', 'softmax_in_fp32': False,
                       'additional_softmax_scale': None, 'softmax_offset': None,
                       'scaled_masked_softmax_fusion': args.backend == 'default-fusion',
                       'kernel_available': None, 'route': None,
                       'forward_completed': False, 'backward_completed': False}
                layer = FusedScaleMaskSoftmax(
                    input_in_fp16=False, input_in_bf16=True, attn_mask_type=AttnMaskType.causal,
                    scaled_masked_softmax_fusion=row['scaled_masked_softmax_fusion'],
                    mask_func=attention_mask_func, softmax_in_fp32=False, scale=None,
                    window_size=None)
                available = layer.is_kernel_available
                fused = layer.forward_fused_softmax
                fallback = layer.forward_torch_softmax

                def observe_available(*positional, **named):
                    row['stage'] = 'native-kernel-availability'
                    result = available(*positional, **named)
                    row['kernel_available'] = bool(result)
                    return result

                def observe_fused(*positional, **named):
                    row.update(stage='native-fused-forward', route='cuda-extension')
                    return fused(*positional, **named)

                def observe_fallback(*positional, **named):
                    row.update(stage='native-torch-forward', route='native-torch-fallback')
                    return fallback(*positional, **named)

                layer.is_kernel_available = observe_available
                layer.forward_fused_softmax = observe_fused
                layer.forward_torch_softmax = observe_fallback
                try:
                    scores_leaf = torch.randn(2, 8, sequence, sequence, device='cuda',
                                              dtype=torch.bfloat16, requires_grad=True)
                    # Native scores come from baddbmm and are not an autograd leaf.
                    # The fallback mask_func writes in place; clone preserves values
                    # while allowing that native operation and its mask gradient.
                    x = scores_leaf.clone()
                    mask = torch.triu(torch.ones(sequence, sequence, device='cuda',
                                                dtype=torch.bool), diagonal=1)
                    mask = mask.view(1, 1, sequence, sequence).expand(2, 1, sequence, sequence).contiguous()
                    before_mask = mask.cpu().clone()
                    y = layer(x, mask, None)
                    row['forward_completed'] = True
                    row['stage'] = 'native-backward'
                    dx = torch.autograd.grad(y, scores_leaf, torch.randn_like(y))[0]
                    torch.cuda.synchronize()
                    row.update(backward_completed=True, output_shape=list(y.shape),
                               output_dtype=str(y.dtype), gradient_shape=list(dx.shape),
                               gradient_dtype=str(dx.dtype),
                               finite_output=bool(torch.isfinite(y.detach().cpu()).all()),
                               finite_gradient=bool(torch.isfinite(dx.detach().cpu()).all()),
                               mask_unchanged=torch.equal(before_mask, mask.cpu()))
                    passed = (y.shape == x.shape and dx.shape == x.shape
                              and y.dtype == dx.dtype == torch.bfloat16
                              and row['finite_output'] and row['finite_gradient'] and row['mask_unchanged'])
                    if args.backend == 'native-fallback':
                        passed = passed and row['route'] == 'native-torch-fallback'
                    row['status'] = 'passed' if passed else 'failed'
                except Exception as error:
                    row.update(status='failed', exception=type(error).__name__, message=str(error),
                               traceback=traceback.format_exc())
                row['cuda_extension_forward_completed'] = (
                    row['route'] == 'cuda-extension' and row['forward_completed'])
                row['cuda_extension_backward_completed'] = (
                    row['route'] == 'cuda-extension' and row['backward_completed'])
                row['extensions_after'] = extension_state()
                receipt['cases'].append(row)
            receipt['status'] = ('passed' if all(row['status'] == 'passed' for row in receipt['cases'])
                                 else 'failed')
    except Exception as error:
        receipt.update(status='failed', exception=type(error).__name__, message=str(error),
                       traceback=traceback.format_exc())
    finally:
        receipt['sources_unchanged'] = all(sha(args.megatron_root/name) == value
                                           for name, value in sources.items())
        assert receipt['sources_unchanged'], 'native sources changed during backend probe'
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps({'status': receipt['status'], 'backend': args.backend,
                      'cases': [{'sequence': row['sequence'], 'status': row['status'],
                                 'route': row['route'], 'kernel_available': row['kernel_available'],
                                 'message': row.get('message')} for row in receipt['cases']]}), flush=True)
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
