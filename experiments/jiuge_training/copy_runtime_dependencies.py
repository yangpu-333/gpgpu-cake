"""Copy pinned package files into the isolated ACT preparation environment.

Run this with the complete source environment's Python. Only files recorded by
importlib.metadata and resolved inside its site-packages are copied. Entry
points outside site-packages are excluded; no GPU calls are made.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shutil
import sys
import sysconfig
import time


PREP = Path('/home/huids25/kda-jiuge-prep-20261014')
SOURCE_ENV = Path('/home/huids25/kda-work/.venv')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def copy_dependencies(requirements: Path, receipt_path: Path) -> dict:
    prep = PREP.resolve(strict=True)
    source = Path(sysconfig.get_paths()['purelib']).resolve(strict=True)
    expected_source = SOURCE_ENV.resolve(strict=True)
    if Path(sys.prefix).resolve(strict=True) != expected_source:
        raise ValueError('Run with the declared, complete source environment')
    if not source.is_relative_to(expected_source):
        raise ValueError('Source site-packages escapes the source environment')
    target = prep / '.venv' / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    target = target.resolve(strict=True)
    if not target.is_relative_to(prep / '.venv') or target == source:
        raise ValueError('Destination must be the isolated preparation environment')
    requirements = requirements.resolve(strict=True)
    receipt_path = receipt_path.resolve()
    if not requirements.is_relative_to(prep) or not receipt_path.is_relative_to(prep):
        raise ValueError('Requirements and receipt must stay within preparation root')
    record = {
        'status': 'running', 'source_site_packages': str(source),
        'target_site_packages': str(target), 'gpu_work_performed': False,
        'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'packages': [],
        'excluded_external_entry_points': [],
    }

    def persist() -> None:
        receipt_path.write_text(json.dumps(record, indent=2) + '\n')

    persist()
    started = time.monotonic()
    try:
        for requirement in requirements.read_text().splitlines():
            requirement = requirement.strip()
            if not requirement or requirement.startswith('#'):
                continue
            name, version = requirement.split('==', 1)
            distribution = metadata.distribution(name)
            if distribution.version != version:
                raise ValueError(f'{name}: source version differs from pinned version')
            package_record = {'name': name, 'version': version, 'files': [], 'bytes': 0}
            for package_file in distribution.files or []:
                declared = Path(os.path.abspath(distribution.locate_file(package_file)))
                origin = declared.resolve(strict=True)
                if not declared.is_relative_to(source) or not origin.is_relative_to(source):
                    record['excluded_external_entry_points'].append(
                        {'package': name, 'file': str(package_file)})
                    continue
                if not origin.is_file():
                    raise ValueError(f'{name}: a metadata entry is not a regular file')
                relative = declared.relative_to(source)
                destination = target / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.parent.resolve(strict=True).is_relative_to(target):
                    raise ValueError('Destination parent escapes target')
                if destination.is_symlink():
                    raise ValueError('Refusing to overwrite a destination symlink')
                source_hash = sha256(origin)
                identical = (destination.is_file()
                             and destination.stat().st_size == origin.stat().st_size
                             and sha256(destination) == source_hash)
                if not identical:
                    temporary = destination.with_name(destination.name + '.kda-copying')
                    if temporary.is_symlink():
                        raise ValueError('Temporary destination is a symlink')
                    shutil.copy2(origin, temporary)
                    if sha256(temporary) != source_hash:
                        raise ValueError(f'Copied content mismatch for {relative}')
                    os.replace(temporary, destination)
                package_record['files'].append({
                    'path': str(relative), 'bytes': origin.stat().st_size,
                    'sha256': source_hash, 'action': 'already_identical' if identical else 'copied',
                })
                package_record['bytes'] += origin.stat().st_size
            record['packages'].append(package_record)
            persist()
            print(json.dumps({'package': name, 'version': version,
                              'files': len(package_record['files']), 'bytes': package_record['bytes']}),
                  flush=True)
        record['status'] = 'complete'
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        record['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        persist()
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--requirements', type=Path, default=PREP / 'torch-deps.txt')
    parser.add_argument('--receipt', type=Path, default=PREP / 'runtime-dependency-copy.json')
    args = parser.parse_args()
    result = copy_dependencies(args.requirements, args.receipt)
    print(json.dumps({'status': result['status'], 'packages': len(result['packages']),
                      'elapsed_seconds': result['elapsed_seconds']}))
