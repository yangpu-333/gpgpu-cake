"""Lossless chunk archive and CPU-float64 comparison for real HDO updates.

The ACT archive is primary. Formal timing remains blocked until the raw archive
is copied and verified locally and all other existing acceptance gates pass.
"""
import hashlib
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import zlib


class FullAudit:
    def __init__(self, campaign, arm, rank):
        self.campaign, self.arm, self.rank = Path(campaign), arm, rank
        self.folder = self.campaign / arm / f'rank-{rank}'
        self.folder.mkdir(parents=True, exist_ok=False)
        self.pack_path = self.folder / 'raw.zlibpack'
        self.pack = self.pack_path.open('xb', buffering=8388608)
        self.local_chunks = {}
        self.compressor = ThreadPoolExecutor(max_workers=4)
        self.step = 0
        self.rows = []
        self.reference = {}
        if arm != 'native':
            source = self.campaign / 'native' / f'rank-{rank}' / 'all-state-manifest.json'
            self.reference = {r['field']: r for r in json.loads(source.read_text())['tensors']}
        self.journal = (self.folder / 'tensor-journal.jsonl').open('x')

    def tensor(self, field, tensor):
        import torch
        value = tensor.detach().reshape(-1)
        row = {'field': field, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
               'numel': tensor.numel(), 'chunks': [], 'passed': True, 'max_abs': 0.0}
        reference = self.reference.get(field)
        if self.arm != 'native' and reference is None:
            raise ValueError('Reference field missing: ' + field)
        if reference and any(row[k] != reference[k] for k in ('shape', 'dtype', 'numel')):
            raise ValueError('Reference structure differs: ' + field)
        pending = deque()
        def drain():
            digest, chunk, future = pending.popleft()
            numel = chunk['numel']
            if future is not None:
                if digest in self.local_chunks:
                    chunk = dict(self.local_chunks[digest])
                else:
                    compressed = future.result()
                    chunk.update(file=str(self.pack_path.relative_to(self.campaign)),
                                 offset=self.pack.tell(), compressed_bytes=len(compressed))
                    self.pack.write(compressed)
                    self.local_chunks[digest] = dict(chunk)
            chunk['numel'] = numel
            row['chunks'].append(chunk)
        for index, start in enumerate(range(0, value.numel(), 8388608)):
            cpu = value[start:start+8388608].cpu().contiguous()
            data = cpu.view(torch.uint8).numpy().tobytes()
            digest = hashlib.sha256(data).hexdigest()
            x = cpu.double()
            if not bool(torch.isfinite(x).all()):
                row['passed'] = False
            if reference:
                old = reference['chunks'][index]
                with (self.campaign / old['file']).open('rb') as stream:
                    stream.seek(old['offset'])
                    raw = zlib.decompress(stream.read(old['compressed_bytes']))
                if hashlib.sha256(raw).hexdigest() != old['sha256']:
                    raise ValueError('Reference raw chunk corrupted')
                y = torch.frombuffer(bytearray(raw), dtype=cpu.dtype).double()
                if x.shape != y.shape:
                    raise ValueError('Reference chunk shape differs')
                row['passed'] &= bool(torch.allclose(x, y, atol=3e-4, rtol=1e-3, equal_nan=False))
                row['max_abs'] = max(row['max_abs'], float((x-y).abs().max()) if x.numel() else 0.0)
            if reference and digest == old['sha256']:
                chunk = dict(old)
                chunk['numel'] = cpu.numel()
                future = None
            elif digest in self.local_chunks:
                chunk = dict(self.local_chunks[digest])
                chunk['numel'] = cpu.numel()
                future = None
            else:
                chunk = {'sha256': digest, 'bytes': len(data), 'numel': cpu.numel()}
                future = self.compressor.submit(zlib.compress, data, 1)
            pending.append((digest, chunk, future))
            if len(pending) >= 8:
                drain()
        while pending:
            drain()
        self.pack.flush()
        self.rows.append(row)
        self.journal.write(json.dumps(row) + '\n')
        self.journal.flush()
        return row

    def save_hdo(self, opt, phase):
        import torch
        torch.cuda.synchronize()
        elements = 0
        count = 0
        for group in opt.param_groups:
            for param in group['params']:
                key = f'step-{self.step}/{count}'
                elements += param.numel()
                if phase == 'before':
                    grad = getattr(param, 'decoupled_grad', None)
                    if grad is None:
                        grad = param.grad
                    if grad is None:
                        raise ValueError('Actual parameter gradient missing')
                    self.tensor(key + '/gradient', grad)
                else:
                    self.tensor(key + '/updated_parameter', param)
                    state = opt.state[param]
                    for field in ('master_param', 'exp_avg', 'exp_avg_sq', 'step'):
                        value = state[field]
                        if not torch.is_tensor(value):
                            value = torch.tensor(value, dtype=torch.float64)
                        self.tensor(key + '/' + field, value)
                count += 1
        if count != 199 or elements != 3834518016:
            raise ValueError(f'Incomplete actual optimizer coverage: {count}, {elements}')
        self.flush()

    def flush(self):
        result = {'schema': 'jiuge7b-pinning-full-numeric-audit-v1', 'arm': self.arm, 'rank': self.rank,
                  'step': self.step, 'comparison_dtype': 'CPU float64', 'atol': 3e-4, 'rtol': 1e-3,
                  'tensors': self.rows, 'passed_so_far': all(r['passed'] for r in self.rows),
                  'reference_fields_expected': len(self.reference), 'local_raw_copy_verified': False,
                  'checkpoint_resume_tested': False, 'formal_performance_passed': False}
        path = self.folder / 'all-state-manifest.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(path)

    def close(self):
        self.compressor.shutdown(wait=True)
        self.pack.close()
        self.journal.close()


def install(campaign, arm, rank, hdo, training):
    import torch
    audit = FullAudit(campaign, arm, rank)
    original = hdo.step
    def observed(opt, *args, **kwargs):
        audit.step += 1
        print(json.dumps({'full_audit_step': audit.step, 'phase': 'before', 'rank': rank}), flush=True)
        audit.save_hdo(opt, 'before')
        result = original(opt, *args, **kwargs)
        audit.save_hdo(opt, 'after')
        print(json.dumps({'full_audit_step': audit.step, 'phase': 'after', 'rank': rank}), flush=True)
        return result
    hdo.step = observed
    train_step = training.train_step
    def with_losses(*args, **kwargs):
        result = train_step(*args, **kwargs)
        for name, value in result[0].items():
            if isinstance(value, (tuple, list)):
                for i, component in enumerate(value):
                    audit.tensor(f'step-{audit.step}/loss/{name}/{i}', torch.as_tensor(component))
            else:
                audit.tensor(f'step-{audit.step}/loss/{name}', torch.as_tensor(value))
        audit.flush()
        return result
    training.train_step = with_losses
    return audit
