"""CPU-only, immutable archive of this day's interrupted memory admissions.

Run on ACT. It never imports Torch, reads CUDA, or downloads large originals.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
OUTPUT = PREP/'sft-direct-zero-failure-preservation-20261023-01'
ROOTS = (
    'sft-direct-zero-first-step-20261023-01',
    'runs/sft-direct-zero-first-step-20261023-01',
    'sft-direct-zero-bootstrap-20261023-01',
    'sft-moment-direct-zero-runtime-sg-huids25-261023-2b50d-default0-0',
    'sft-direct-zero-first-step-20261023-02',
    'runs/sft-direct-zero-first-step-20261023-02',
    'sft-direct-zero-bootstrap-20261023-02',
    'sft-moment-direct-zero-runtime-sg-huids25-261023-f53a4-default0-0',
    'node-memory-capacity-20261023-01',
    'sft-phased-moment-preparation-20261023-03',
)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=True)
    manifest = OUTPUT/'manifest.json'
    receipt = OUTPUT/'verification.json'
    bundle = OUTPUT/'Jiuge7B-SFT-direct-zero-interrupted-and-phased-CPU-evidence.zip'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        assert digest(bundle) == result['bundle_sha256']
        assert digest(manifest) == result['manifest_sha256']
        print(json.dumps(result), flush=True)
        return
    if manifest.exists() or bundle.exists():
        raise FileExistsError('Incomplete preservation exists; inspect before versioned recovery')
    paths = {}
    for relative in ROOTS:
        root = PREP/relative
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob('*')):
            if '__pycache__' in path.parts or path.is_dir():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError('Do not archive symlink or special file: '+str(path))
            name = path.relative_to(PREP).as_posix()
            paths[name] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    metadata = {
        'schema': 'jiuge7b-interrupted-memory-admission-immutable-archive-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'roots': list(ROOTS), 'files': paths,
        'GPU_used_for_preservation': False, 'originals_modified': False,
        'first_real_update_completed': False, 'six_case_screen_started': False,
        'new_7B_correctness_or_performance_accepted': False,
    }
    with manifest.open('x') as stream:
        json.dump(metadata, stream, indent=2)
    with zipfile.ZipFile(bundle, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3, allowZip64=True) as archive:
        for name in paths:
            archive.write(PREP/name, name)
        archive.write(manifest, 'preservation-manifest.json')
    with zipfile.ZipFile(bundle) as archive:
        assert len(archive.namelist()) == len(paths)+1
        for name, expected in paths.items():
            actual = hashlib.sha256()
            length = 0
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    actual.update(block)
                    length += len(block)
            assert length == expected['bytes'] and actual.hexdigest() == expected['sha256'], name
            assert digest(PREP/name) == expected['sha256'], name
        assert archive.read('preservation-manifest.json') == manifest.read_bytes()
    result = {
        'schema': 'jiuge7b-interrupted-memory-admission-preservation-receipt-v1',
        'passed': True, 'ACT_bundle': str(bundle), 'bundle_bytes': bundle.stat().st_size,
        'bundle_sha256': digest(bundle), 'manifest_sha256': digest(manifest),
        'verified_files': len(paths), 'original_bytes': sum(v['bytes'] for v in paths.values()),
        'compressed_and_decompressed_SHA_passed': True,
        'raw_originals_downloaded_to_local': False, 'GPU_used': False,
        'originals_modified': False,
    }
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
