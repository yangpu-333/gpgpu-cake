"""Versioned observer recovery; preserve all158 original scientific sources."""
import json
import os
from pathlib import Path

import sft_phased_acceptance_20261024 as core

OLD = core.PREP/'sft-phased-acceptance-20261024-01'


def configure():
    core.ROOT = core.PREP/'sft-phased-acceptance-20261024-02'
    core.APPROVAL = core.PREP/'sft-phased-acceptance-approved-20261024-02.json'
    core.MANIFEST = 'sft-phased-acceptance-source-manifest-20261024-02.json'
    return core


def alias_completed(numeric, semantics, state, cases):
    prior = core.read(OLD/'controller.json')
    expected = core.read(core.PREP/'sft-phased-acceptance-source-manifest-20261024-01.json')['files']
    if (prior['status']!='stopped' or prior['formal_performance_started']
            or prior['sources']!=expected
            or prior.get('error')!='phased-audit-optimized-20261024-01 failed; no subsequent GPU stage admitted'
            or prior['completed_cases']!=['phased-audit-native-20261024-01','phased-audit-control-20261024-01']):
        raise ValueError('Only the exact completed same-campaign baseline/control may be reused')
    if prior['approval']['ordered_gpu_uuids']!=core.read(core.APPROVAL)['ordered_gpu_uuids']:
        raise ValueError('Preserve original physical rank/GPU pair across resource retry')
    for name in ('verify-source-native','verify-source-control'):
        rows=[x for x in prior['commands'] if x['name']==name]
        if len(rows)!=1 or rows[0]['exit_code']:
            raise ValueError('Both completed SFT source/mask reconstructions required')
    for name,digest in expected.items():
        if core.sha(core.PREP/name)!=digest or core.sha(OLD/'source'/name)!=digest:
            raise ValueError('Original158 byte identities changed')
    origins = []
    def link(source, destination):
        source = source.resolve(strict=True)
        if not source.is_relative_to(OLD.resolve()) or not source.is_file() or source.is_symlink():
            raise ValueError('Only immutable owned reference files may be aliased')
        destination.parent.mkdir(parents=True,exist_ok=True)
        os.link(source,destination)
        a,b=source.stat(),destination.stat()
        if (a.st_dev,a.st_ino,a.st_size)!=(b.st_dev,b.st_ino,b.st_size):
            raise ValueError('Reference alias must be identical inode/bytes')
        origins.append({'source':str(source),'alias':str(destination),'bytes':a.st_size,'same_inode':True})
    for arm in ('native','control'):
        for rank in (0,1):
            folder = OLD/'numeric'/arm/f'rank-{rank}'
            m = core.read(folder/'all-state-manifest.json')
            done = core.read(folder/'completion.json')
            if (not done['passed'] or not done['complete_fields'] or done['steps']!=3
                    or len(m['tensors'])!=3585 or not all(x['passed'] for x in m['tensors'])
                    or m['atol']!=3e-4 or m['rtol']!=1e-3):
                raise ValueError('Completed complete three-step reference required')
            for path in folder.rglob('*'):
                if path.is_file():
                    link(path,numeric/path.relative_to(OLD/'numeric'))
        for path in (OLD/'semantics'/('semantics-'+arm)).rglob('*'):
            if path.is_file():
                link(path,semantics/path.relative_to(OLD/'semantics'))
        case = core.PREP/'runs'/f'phased-audit-{arm}-20261024-01'
        record = core.read(case/'case.json')
        if record['exit_code'] or record['resource_gate_before'].strip() or record['resource_gate_after'].strip():
            raise ValueError('Reference GPU occupancy/exit gate failed')
        cases.append(case)
        state['completed_cases'].append(record['case'])
    with (numeric.parent/'reference-origins.json').open('x') as stream:
        json.dump({'same_campaign_references_not_new_runs':True,'old158_sources_unchanged':True,
                   'candidate_failed_attempt_not_reused':True,'performance_results_reused':False,
                   'original_controller_sha256':core.sha(OLD/'controller.json'),'files':origins},stream,indent=2)
