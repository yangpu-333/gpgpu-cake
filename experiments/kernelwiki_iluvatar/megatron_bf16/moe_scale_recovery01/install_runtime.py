"""Restore only exact archive files into fixed empty or identical owned targets."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import socket
import zipfile

REMOTE=Path('/private/kda-biv150-stage35-recovery01')
TARGETS={'native':Path('/tmp/kda-bf16-continuation/megatron-native-5be9626'),
    'bf16':Path('/tmp/kda-bf16-continuation/megatron_bf16'),
    'formal':Path('/tmp/kda-bf16-continuation/megatron_training_entry'),
    'gradnorm':Path('/tmp/kda-gradnorm-stage26'),'experiment':REMOTE}
sha=lambda data:hashlib.sha256(data).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--sha256',required=True);args=parser.parse_args()
    output=REMOTE/'runtime-restored-01.json';assert not output.exists()
    assert Path('/private').resolve()==Path('/mnt') and sha(args.archive.read_bytes())==args.sha256
    with zipfile.ZipFile(args.archive) as archive:
        manifest=json.loads(archive.read('manifest.json'));rows=manifest['files']
        assert set(archive.namelist())=={'manifest.json',*(row['path'] for row in rows)}
        assert manifest['native_sources']==600 and manifest['model_controller_files_byte_identical']==25
        checked=[]
        for row in rows:
            relative=PurePosixPath(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] in TARGETS
            root=TARGETS[relative.parts[0]];path=root.joinpath(*relative.parts[1:])
            assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
            data=archive.read(row['path']);assert len(data)==row['bytes'] and sha(data)==row['sha256']
            if path.exists():assert path.is_file() and path.read_bytes()==data
            else:
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write(data)
            checked.append(dict(row,remote_path=str(path)))
    result={'schema':'stage35-recovery01-runtime-restoration-v1','hostname':socket.gethostname(),
        'status':'restored_exact_bytes','archive_sha256':args.sha256,'files':checked,
        'native_sources':600,'model_controller_files_byte_identical':25,
        'raw_storage':str(REMOTE.resolve()),'installed_packages_copied':False,'gpu_work_executed':False}
    with output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:result[key] for key in ('status','hostname','native_sources','model_controller_files_byte_identical','raw_storage')}))


if __name__=='__main__':main()
