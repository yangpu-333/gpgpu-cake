"""Stage official checkpoint members with bounded memory and resumable ranges.

The metadata manifest must come from the inspected public tar headers. This
downloads model files only; it never imports model code or allocates tensors.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.error
import urllib.request

OFFICIAL_URL = 'https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9g_8b_v2_thinking.tar'
OFFICIAL_SOURCES = {
    OFFICIAL_URL: '9g_8b_thinking',
    'https://thunlp-model.oss-cn-wulanchabu.aliyuncs.com/9G7B_MHA.tar': '9G7B_MHA',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def member_path(root, member, expected_directory='9g_8b_thinking'):
    relative = PurePosixPath(member['name'])
    if not relative.parts or relative.is_absolute() or relative.parts[0] != expected_directory or '..' in relative.parts:
        raise ValueError('Unsafe or unexpected checkpoint member path')
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Member path escaped checkpoint root')
    return target


def stage(manifest, root, receipt_path):
    url = manifest.get('url')
    if url not in OFFICIAL_SOURCES or manifest.get('status') != 'metadata_read':
        raise ValueError('Inspected official metadata manifest required')
    expected_directory = OFFICIAL_SOURCES[url]
    request = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = int(response.headers['Content-Length'])
        etag = response.headers['ETag']
    if archive_bytes != manifest['archive_bytes']:
        raise ValueError('Archive size changed since metadata inspection')
    if manifest.get('etag') is not None and manifest['etag'] != etag:
        raise ValueError('Archive ETag changed since metadata inspection')
    members = [m for m in manifest['members'] if m['bytes']]
    targets = [(m, member_path(root, m, expected_directory)) for m in members]
    if len({str(p) for _, p in targets}) != len(targets):
        raise ValueError('Duplicate destination path')
    root.mkdir(parents=True, exist_ok=True)
    if receipt_path.exists():
        record = json.loads(receipt_path.read_text())
        if record['url'] != url or record['etag'] != etag or record['archive_bytes'] != archive_bytes:
            raise ValueError('Source identity differs from previous partial download')
        if 'error' in record:
            record.setdefault('previous_failures', []).append({
                'error_type': record.pop('error_type', None), 'error': record.pop('error'),
                'recorded_utc': record.get('updated_utc')})
    else:
        record = {'schema': 'jiuge-checkpoint-staging-v1', 'url': url,
                  'archive_bytes': archive_bytes, 'etag': etag,
                  'started_utc': datetime.now(timezone.utc).isoformat(),
                  'code_executed': False, 'gpu_allocated': False, 'files': {}}
        record['member_selection_policy'] = manifest.get('selection_policy', 'unique member paths required')
        record['omitted_duplicate_entries'] = manifest.get('omitted_duplicate_entries', [])

    def save(status):
        record['status'] = status
        record['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        temporary.replace(receipt_path)

    save('downloading')
    try:
        for member, target in targets:
            key, size = member['name'], member['bytes']
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                prior = record['files'].get(key, {})
                if target.stat().st_size != size or prior.get('sha256') != digest(target):
                    raise ValueError('Existing checkpoint file lacks a matching receipt: ' + key)
                continue
            partial = target.with_name(target.name + '.download')
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ValueError('Partial file larger than the source member')
            remaining = sum(m['bytes'] - (p.stat().st_size if p.exists() else
                            (p.with_name(p.name + '.download').stat().st_size
                             if p.with_name(p.name + '.download').exists() else 0))
                            for m, p in targets)
            if shutil.disk_usage(root).free < remaining + 1024 ** 3:
                raise ValueError('Insufficient filesystem capacity for remaining checkpoint members')
            retries = 0
            last_report = time.monotonic()
            while offset < size:
                start = member['data_offset'] + offset
                end_offset = min(size, offset + 256 * 1024 * 1024)
                end = member['data_offset'] + end_offset - 1
                req = urllib.request.Request(url, headers={
                    'Range': f'bytes={start}-{end}', 'If-Match': etag})
                try:
                    with urllib.request.urlopen(req, timeout=40) as response:
                        expected_range = f'bytes {start}-{end}/{archive_bytes}'
                        if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                            raise ValueError('Exact HTTP range response required before reading payload')
                        if response.headers.get('ETag') != etag:
                            raise ValueError('Archive ETag changed during staging')
                        with partial.open('ab') as output:
                            while offset < end_offset:
                                chunk = response.read(min(4 * 1024 * 1024, end_offset - offset))
                                if not chunk:
                                    raise EOFError('Source response ended before complete range; partial retained')
                                output.write(chunk)
                                offset += len(chunk)
                                if time.monotonic() - last_report >= 10:
                                    output.flush()
                                    record['current_member'] = {'name': key, 'received_bytes': offset, 'bytes': size}
                                    save('downloading')
                                    print(json.dumps(record['current_member']), flush=True)
                                    last_report = time.monotonic()
                            output.flush()
                            os.fsync(output.fileno())
                except (TimeoutError, urllib.error.URLError, EOFError, ConnectionError,
                        http.client.HTTPException) as error:
                    if isinstance(error, urllib.error.HTTPError) and error.code in (401, 403, 404, 412, 416):
                        raise
                    retries += 1
                    if retries > 8:
                        raise
                    offset = partial.stat().st_size if partial.exists() else 0
                    record.setdefault('network_retries', []).append({
                        'member': key, 'offset': offset, 'error_type': type(error).__name__})
                    save('downloading')
                    print(json.dumps({'retry': retries, 'member': key, 'offset': offset,
                                      'error_type': type(error).__name__}), flush=True)
                    time.sleep(min(10, retries * 2))
            actual = digest(partial)
            expected = manifest.get('metadata_files', {}).get(target.name, {}).get('sha256')
            if expected and actual != expected:
                raise ValueError('Inspected metadata bytes differ from the complete download: ' + key)
            partial.replace(target)
            record['files'][key] = {'bytes': size, 'sha256': actual, 'path': str(target)}
            record.pop('current_member', None)
            save('downloading')
            print(json.dumps({'member_complete': key, 'bytes': size, 'sha256': actual}), flush=True)
        record['file_count'] = len(members)
        record['member_bytes'] = sum(m['bytes'] for m in members)
        record['checkpoint_directory'] = str(root / expected_directory)
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save('complete')
        print(json.dumps({'status': 'complete', 'directory': record['checkpoint_directory'],
                          'files': record['file_count'], 'bytes': record['member_bytes']}), flush=True)
    except Exception as error:
        record['error_type'], record['error'] = type(error).__name__, str(error)
        save('incomplete_resumable')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    stage(json.loads(args.manifest.read_text()), args.root, args.receipt)


if __name__ == '__main__':
    main()
