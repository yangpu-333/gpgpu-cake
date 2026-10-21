"""Verify complete new affinity numeric/resume originals on ACT."""
import argparse
import hashlib
import json
from pathlib import Path
import zlib
from verify_pinning_archive import validate_coverage, WIDTHS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); root=args.root.resolve(strict=True)
    checked=set(); rows=[]
    references={rank:validate_coverage(json.loads((root/'native'/f'rank-{rank}'/'all-state-manifest.json').read_text()))
                for rank in (0,1)}
    for arm in ('optimized','resume-restored','resume-next-step'):
        for rank in (0,1):
            folder=root/arm/f'rank-{rank}'
            path=folder/'all-state-manifest.json'; manifest=json.loads(path.read_text())
            actual={row['field']:row for row in manifest['tensors']}
            if len(actual)!=len(manifest['tensors']):
                raise ValueError('Duplicate actual field')
            if arm=='optimized':
                expected=references[rank]
                validate_coverage(manifest)
                proof=json.loads((folder/'completion.json').read_text())
                if not proof['passed'] or not proof['complete_fields'] or proof['steps']!=3:
                    raise ValueError('Complete three-step new candidate required')
            else:
                step=2 if arm=='resume-restored' else 3
                expected={name:row for name,row in references[rank].items() if name.startswith(f'step-{step}/')}
                if step==2:
                    expected={name:row for name,row in expected.items() if not name.endswith('/gradient') and '/loss/' not in name}
                if not json.loads((root/f'resume-rank-{rank}-completion.json').read_text())['passed']:
                    raise ValueError('New full restore/next-update comparison required')
            if actual.keys()!=expected.keys():
                raise ValueError('Complete field coverage differs')
            for name,row in actual.items():
                reference=references[rank][name]
                if not row['passed'] or any(row[k]!=reference[k] for k in ('shape','numel','dtype')):
                    raise ValueError('Numerical comparison or structure differs')
                if sum(chunk['numel'] for chunk in row['chunks'])!=row['numel']:
                    raise ValueError('Chunk extent incomplete')
                for chunk in row['chunks']:
                    path=(root/chunk['file']).resolve(strict=True)
                    if not path.is_relative_to(root) or chunk['bytes']!=chunk['numel']*WIDTHS[row['dtype']]:
                        raise ValueError('Unsafe or incomplete raw chunk')
                    key=(str(path),chunk['offset'],chunk['compressed_bytes'],chunk['sha256'],chunk['bytes'])
                    if key in checked:
                        continue
                    with path.open('rb') as stream:
                        stream.seek(chunk['offset']); raw=zlib.decompress(stream.read(chunk['compressed_bytes']))
                    if len(raw)!=chunk['bytes'] or hashlib.sha256(raw).hexdigest()!=chunk['sha256']:
                        raise ValueError('ACT raw numeric original corrupted')
                    checked.add(key)
            rows.append({'arm':arm,'rank':rank,'fields':len(actual),'max_abs_error':max(row['max_abs'] for row in actual.values()),
                         'manifest_sha256':hashlib.sha256((folder/'all-state-manifest.json').read_bytes()).hexdigest()})
    result={'schema':'jiuge7b-cpu-numa-full-archive-verification-v1','passed':True,'root':str(root),
            'new_candidate_and_resume_runs':rows,'unique_raw_chunks_verified':len(checked),
            'primary_complete_raw_values_verified':True,'full_raw_local_copy_verified':False,
            'baseline_reference_origin':json.loads((root/'reference-origins.json').read_text()),
            'formal_performance_passed':False}
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key!='baseline_reference_origin'}),flush=True)


if __name__=='__main__':
    main()
