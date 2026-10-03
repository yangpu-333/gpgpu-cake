"""Inventory preserved raw snapshots, including checkpoints, without unpickling."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence_root',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    files = []
    for path in sorted(args.evidence_root.rglob('*')):
        if path.is_file() and 'snapshots' in path.relative_to(args.evidence_root).parts:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(block)
            files.append({'relative_path':path.relative_to(args.evidence_root).as_posix(),
                          'bytes':path.stat().st_size, 'sha256':digest.hexdigest()})
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump({'raw_root':str(args.evidence_root.resolve()), 'files':files,
            'count':len(files), 'total_bytes':sum(row['bytes'] for row in files),
            'scope':'Original tensors and native checkpoints preserved separately; binary data excluded from Git, receipts included.'},stream,indent=2)
        stream.write('\n')
    print(len(files), 'raw files',sum(row['bytes'] for row in files),'bytes')


if __name__ == '__main__':
    main()
