"""Verify all lossless real-7B audit values before admitting formal timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import zlib

FIELDS = ('gradient', 'updated_parameter', 'master_param', 'exp_avg', 'exp_avg_sq', 'step')
WIDTHS = {'torch.bfloat16': 2, 'torch.float32': 4, 'torch.float64': 8,
          'torch.int32': 4, 'torch.int64': 8, 'torch.bool': 1}


def validate_coverage(manifest):
    tensors = manifest['tensors']
    actual = {r['field']: r for r in tensors}
    if len(actual) != len(tensors):
        raise ValueError('Duplicate full-state fields')
    expected = {f'step-{step}/{index}/{field}' for step in (1, 2, 3)
                for index in range(199) for field in FIELDS}
    non_loss = {name for name in actual if '/loss/' not in name}
    if non_loss != expected:
        raise ValueError('Incomplete parameter/gradient/master/moment/step coverage')
    for step in (1, 2, 3):
        prefix = f'step-{step}/'
        if not any(name.startswith(prefix + 'loss/') for name in actual):
            raise ValueError('Actual full-precision loss missing')
        for field in FIELDS[:-1]:
            if sum(actual[f'{prefix}{i}/{field}']['numel'] for i in range(199)) != 3834518016:
                raise ValueError('Actual full parameter element coverage differs')
    return actual


def verify(root, include_resume=False):
    root = root.resolve(strict=True)
    checked = set()
    rows = []
    references = {}
    for arm in ('native', 'control', 'optimized'):
        for rank in (0, 1):
            folder = root / arm / f'rank-{rank}'
            completion = json.loads((folder / 'completion.json').read_text())
            if not completion['passed'] or not completion['complete_fields'] or completion['steps'] != 3:
                raise ValueError(f'Incomplete or failed numerical run: {arm}/{rank}')
            manifest = json.loads((folder / 'all-state-manifest.json').read_text())
            fields = validate_coverage(manifest)
            if arm == 'native':
                references[rank] = fields
            elif fields.keys() != references[rank].keys():
                raise ValueError('Reference field coverage differs')
            for row in manifest['tensors']:
                if not row['passed']:
                    raise ValueError('Numerical comparison failed')
                if arm != 'native' and any(row[k] != references[rank][row['field']][k] for k in ('shape', 'numel', 'dtype')):
                    raise ValueError('Raw reference structure differs')
                width = WIDTHS[row['dtype']]
                if sum(c['numel'] for c in row['chunks']) != row['numel'] or any(c['bytes'] != c['numel'] * width for c in row['chunks']):
                    raise ValueError('Raw chunk extent does not cover the tensor')
                for chunk in row['chunks']:
                    path = (root / chunk['file']).resolve(strict=True)
                    if not path.is_relative_to(root):
                        raise ValueError('Raw file escapes archive')
                    key = (str(path.relative_to(root)), chunk['offset'], chunk['compressed_bytes'], chunk['sha256'], chunk['bytes'])
                    if key in checked:
                        continue
                    with path.open('rb') as stream:
                        stream.seek(chunk['offset'])
                        raw = zlib.decompress(stream.read(chunk['compressed_bytes']))
                    if len(raw) != chunk['bytes'] or hashlib.sha256(raw).hexdigest() != chunk['sha256']:
                        raise ValueError('Raw numeric archive differs')
                    checked.add(key)
            rows.append({'arm': arm, 'rank': rank, 'field_count': len(fields),
                         'max_abs_error': max(r['max_abs'] for r in manifest['tensors']),
                         'manifest_sha256': hashlib.sha256((folder / 'all-state-manifest.json').read_bytes()).hexdigest()})
    if include_resume:
        for rank in (0, 1):
            receipt = json.loads((root / f'resume-rank-{rank}-completion.json').read_text())
            if not receipt['passed']:
                raise ValueError('Native checkpoint resume did not pass')
            for arm, step in (('resume-restored', 2), ('resume-next-step', 3)):
                folder = root / arm / f'rank-{rank}'
                manifest = json.loads((folder / 'all-state-manifest.json').read_text())
                actual = {r['field']: r for r in manifest['tensors']}
                expected = {name for name in references[rank] if name.startswith(f'step-{step}/')}
                if step == 2:
                    expected = {name for name in expected if not name.endswith('/gradient') and '/loss/' not in name}
                if len(actual) != len(manifest['tensors']) or actual.keys() != expected:
                    raise ValueError('Full restored/resumed state coverage missing')
                for row in manifest['tensors']:
                    if not row['passed']:
                        raise ValueError('Resume numerical difference')
                    reference = references[rank][row['field']]
                    if any(row[k] != reference[k] for k in ('shape', 'numel', 'dtype')):
                        raise ValueError('Restored state structure differs')
                    if sum(c['numel'] for c in row['chunks']) != row['numel'] or any(c['bytes'] != c['numel'] * WIDTHS[row['dtype']] for c in row['chunks']):
                        raise ValueError('Restored raw extent differs')
                    for chunk in row['chunks']:
                        path = (root / chunk['file']).resolve(strict=True)
                        if not path.is_relative_to(root):
                            raise ValueError('Restored raw path escapes archive')
                        key = (str(path.relative_to(root)), chunk['offset'], chunk['compressed_bytes'], chunk['sha256'], chunk['bytes'])
                        if key not in checked:
                            with path.open('rb') as stream:
                                stream.seek(chunk['offset'])
                                raw = zlib.decompress(stream.read(chunk['compressed_bytes']))
                            if len(raw) != chunk['bytes'] or hashlib.sha256(raw).hexdigest() != chunk['sha256']:
                                raise ValueError('Restored raw original differs')
                            checked.add(key)
                rows.append({'arm': arm, 'rank': rank, 'field_count': len(actual),
                             'max_abs_error': max(r['max_abs'] for r in manifest['tensors']),
                             'manifest_sha256': hashlib.sha256((folder / 'all-state-manifest.json').read_bytes()).hexdigest()})
    return {'schema': 'jiuge7b-pinning-full-archive-verification-v1', 'root': str(root),
            'hostname': socket.gethostname(), 'os_name': os.name, 'runs': rows,
            'unique_raw_chunks_verified': len(checked), 'passed': True,
            'primary_complete_raw_values_verified': True,
            'storage_policy': 'Complete numeric originals at ACT; local key results only',
            'local_windows_raw_copy_verified': os.name == 'nt',
            'checkpoint_resume_tested': include_resume, 'formal_performance_passed': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-resume', action='store_true')
    args = parser.parse_args()
    result = verify(args.root, args.include_resume)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)
