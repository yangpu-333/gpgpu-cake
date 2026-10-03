"""Freeze native training source bytes from the recorded Git commit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

COMMIT = '5be9626709af2722333bf54797c954c09edeada3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite source manifest')
    tree = subprocess.check_output(['git', '-C', str(args.repository), 'ls-tree', '-rz', COMMIT])
    selected = []
    for entry in tree.split(b'\0'):
        if not entry:
            continue
        metadata, name = entry.split(b'\t', 1)
        _, kind, object_id = metadata.split()
        name = name.decode('utf-8')
        path = Path(name)
        if kind == b'blob' and ((name.startswith('megatron/') and path.suffix in ('.py', '.cpp', '.h', '.cu'))
                or (len(path.parts) == 1 and (path.suffix == '.py' or name == 'pyproject.toml'))):
            selected.append((name, object_id))
    # Read raw objects: git archive can apply core.autocrlf on Windows.
    batch = subprocess.check_output(['git', '-C', str(args.repository), 'cat-file', '--batch'],
                                    input=b''.join(oid+b'\n' for _,oid in selected))
    stream = io.BytesIO(batch)
    files = {}
    for name, object_id in selected:
        oid, kind, length = stream.readline().split()
        if oid != object_id or kind != b'blob':
            raise RuntimeError('unexpected Git object response')
        payload = stream.read(int(length))
        if stream.read(1) != b'\n':
            raise RuntimeError('invalid Git batch framing')
        files[name] = hashlib.sha256(payload).hexdigest()
    args.output.write_text(json.dumps({'commit': COMMIT,
        'source': 'Megatron-LM raw Git blobs via cat-file --batch; exact bytes without Windows archive/worktree conversion',
        'files': dict(sorted(files.items()))}, indent=2)+'\n', encoding='utf-8')
    print('Frozen native source files:', len(files))


if __name__ == '__main__':
    main()
