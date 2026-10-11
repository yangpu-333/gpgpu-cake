"""CPU receipt validation of frozen stage35 MoE heldout experiments.
No Torch/Triton/candidate/controller execution; original raw coverage remains explicit.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib,importlib.util,json,math,re,statistics
from pathlib import Path,PurePosixPath
HERE=Path(__file__).resolve().parent
BF=HERE.parent
FORMAL=BF.parent/'megatron_training_entry'
REMOTE='/private/kda-biv150-stage35-recovery01/models/'
CANDIDATE_SHA='b9a2ca76c294481e38e2422332ccb821224bd64b8bb7cd61d972a5e2a4a9133e'
CONTROLLERS_SHA='533af8c2a305d0c92f03c585505e60d16a47c6998f9f5c8b618d368cb222ec10'
MODELS={'long':{'parameters':55718400,'tensors':56},'experts':{'parameters':106189312,'tensors':88}}
ORDERS=[['native','control','optimized'],['optimized','control','native'],['control','native','optimized']]
def require(value, message):
    if not value:
        raise ValueError(message)


SHA_CACHE = {}

def sha(path):
    path=Path(path)
    stamp=path.stat()
    key=(stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns)
    if key in SHA_CACHE:
        return SHA_CACHE[key]
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    value=h.hexdigest()
    after=path.stat()
    require((after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)==key,"original file changed during hashing")
    SHA_CACHE[key]=value
    return value


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def close(left, right):
    require(math.isfinite(left) and math.isfinite(right)
            and math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12), 'statistics differ')


def logged_losses(log):
    """Select the twelve ten-step reports; the separate iteration1 row is excluded."""
    rows=[]
    for line in log.splitlines():
        iteration=re.search(r'\biteration\s+(\d+)/\s*120\s*\|',line)
        if iteration is None or int(iteration[1])==1:
            continue
        lm=re.search(r'\blm loss:\s*([0-9.+eE-]+)',line)
        router=re.search(r'\bload_balancing_loss:\s*([0-9.+eE-]+)',line)
        require(lm is not None and router is not None,'original logged loss fields missing')
        require(math.isfinite(float(lm[1])) and math.isfinite(float(router[1])),'nonfinite logged loss')
        rows.append({'iteration':int(iteration[1]),'lm_loss_text':lm[1],'router_loss_text':router[1]})
    require([row['iteration'] for row in rows]==list(range(10,121,10)),
            'all12 original loss intervals required')
    return rows


def local_original(value,base):
    if value.startswith(REMOTE):
        rel=PurePosixPath(value[len(REMOTE):]);require(not rel.is_absolute() and '..' not in rel.parts,'unsafe raw path')
        require(rel.parts and rel.parts[0] in MODELS,'unexpected heldout model path')
        for folder in (HERE/'models'/rel.parts[0]/'evidence',HERE/'models'/rel.parts[0],HERE/'evidence/raw-backup'/rel.parts[0]):
            path=folder.joinpath(*rel.parts[1:])
            if path.is_file():return path
        return None
    roots={'/tmp/kda-bf16-continuation/megatron_bf16/':BF,'/tmp/kda-bf16-continuation/megatron_training_entry/':FORMAL,'/tmp/kda-gradnorm-stage26/':BF/'evidence/gradnorm-stage26'}
    for remote,local in roots.items():
        if value.startswith(remote):
            rel=PurePosixPath(value[len(remote):]);require(not rel.is_absolute() and '..' not in rel.parts,'unsafe dependency path')
            path=local.joinpath(*rel.parts);return path if path.is_file() else None
    raise ValueError('unexpected original path: '+value)


def verify(model,base=None,require_raw=False,confirmation=False,audit_only=False):
    root=HERE/'models'/model;version='004';base=Path(base) if base is not None else root/'evidence'
    path=HERE/'models/controllers.json';require(sha(path)==CONTROLLERS_SHA,'frozen controllers manifest differs')
    frozen=read(path)
    for name,digest in frozen['files'].items():require(sha(HERE/'models'/name)==digest,'frozen controller changed: '+name)
    candidate = root/'candidates'/version/'candidate.py'
    require(sha(candidate)==CANDIDATE_SHA,'heldout candidate was retuned')
    generation = read(candidate.with_name('generation.json'))
    require(generation['candidate_sha256'] == sha(candidate), 'generated candidate differs')
    response = BF/'evidence'/('cc-moe-weighted-'+version+'.json')
    require(generation['response_sha256'] == sha(response) and read(response)['is_error'] is False,
            'original Claude Code generation missing')
    contract = read(root/'contract.json')
    P=contract['parameter_tensor_count'];L=contract['model']['num_layers'];E=contract['model']['num_moe_experts']
    W=L*E*120
    GROUPS={'initial':P,'actual_start_model_parameters':P,'actual_start_master_parameters':P,'master_initialization':2*P,'model_initialization':2*P,'batches':15,'output':3,'derived_loss':3,'wgrads':3*P,'params':3*P,'masters':3*P,'master_model_consistency':6*P,'routing':9*L}
    require(contract['model']['parameters']==MODELS[model]['parameters'] and P==MODELS[model]['tensors'],'declared heldout model differs')
    require(contract['correctness']['atol'] == 3e-4 and contract['correctness']['rtol'] == 1e-3,
            'frozen tolerance changed')
    audit = read(base/('audit-'+version+'.json'))
    require(audit['passed'] is True and audit['candidate_sha256'] == sha(candidate)
            and audit['contract_sha256'] == sha(root/'contract.json'), 'sealed audit differs')
    for name, digest in audit['controller_sha256'].items():
        require(frozen['files'][model+'/'+name] == digest == sha(root/name), 'audit controller changed')
    require(audit['replay_sha256']==sha(base/'diagnostics-004.json') and audit['replay_controller_sha256']==sha(root/'replay.py'),'real replay identity differs')
    require(sha(HERE/'plan.json')==frozen['plan_sha256'] and sha(HERE/'provenance.json')==frozen['provenance_sha256'],'predeclared plan differs')
    backend_path=HERE/'models/long/evidence/backend-current.json'
    backend=read(backend_path);orchestration=read(base/'audit-orchestration.json')
    require(orchestration['status']=='passed' and orchestration['model']==model
            and orchestration['controllers_sha256']==CONTROLLERS_SHA
            and orchestration['backend_sha256']==sha(backend_path)
            and all(row['exit_code']==0 for row in orchestration['completed']),
            'original audit orchestration differs')
    require(backend['passed'] is True and backend['gpu']=='Iluvatar BI-V150'
            and backend['script_sha256']==sha(HERE/'models/long/backend.py')
            and backend['allocation_verified'] is True
            and backend['source_manifest_sha256']==sha(FORMAL/'source_manifest.json')
            and backend['native_sources_sha256']==read(FORMAL/'source_manifest.json')['files']
            and len(backend['sort_index_probes'])==40
            and all(row['bitwise_equal'] is True for row in backend['sort_index_probes'])
            and {(r['count'],r['pattern'],r['descending']) for r in backend['sort_index_probes']}==
                {(n,p,d) for n in (0,1,17,2048,8192) for p in ('zero','one','random','skewed') for d in (False,True)},
            'fresh native backend and sort probes differ')
    raw_rows = {}
    comparisons = {}
    for name, arms in [('comparison-admission01.json', ('native', 'control')),
                       ('comparison-native-'+version+'.json', ('native', 'optimized')),
                       ('comparison-control-'+version+'.json', ('control', 'optimized'))]:
        value = read(base/name)
        require(audit['comparisons'][name]['sha256'] == sha(base/name), 'comparison bytes differ')
        require(value['passed'] is True and (value['left_arm'], value['right_arm']) == arms
                and value['atol'] == 3e-4 and value['rtol'] == 1e-3
                and value['comparison_device'] == 'CPU float64 copies', 'original numerical gate differs')
        require(value['comparator_sha256']=='feb475eb504fe452267f930f2069d5f17b36387a6cb4ad0f839a6aaa8df27eda' and value['stage35_comparator_sha256']==sha(root/'compare.py'),'frozen numerical comparator identity differs')
        checks, tensors = value['checks'], value['tensor_comparisons']
        require(len(checks)>0 and all(r['passed'] is True for r in checks)
                and len(tensors) == sum(GROUPS.values()) and dict(Counter(r['group'] for r in tensors)) == GROUPS
                and len({r['path'] for r in tensors}) == len(tensors), 'numerical coverage differs')
        for row in tensors:
            require(row['shape_dtype_layout_match'] is row['all_elements_finite'] is row['passed'] is True
                    and row['elements_outside_tolerance'] == 0, 'failed numerical tensor')
            require(row['atol'] == (0 if row['exact_required'] else 3e-4)
                    and row['rtol'] == (0 if row['exact_required'] else 1e-3), 'per-tensor gate changed')
            if row['exact_equal']:
                require(row['native_sha256'] == row['optimized_sha256']
                        and row['elements_not_exactly_equal'] == row['max_absolute_error'] == 0,
                        'exact tensor receipt inconsistent')
        require(len(value['input_files']) == 35 and len(value['route_input_files']) == 48,
                'original comparison inputs missing')
        for row in value['input_files'] + value['route_input_files']:
            previous = raw_rows.setdefault(row['path'], row)
            require(previous['bytes'] == row['bytes'] and previous['sha256'] == row['sha256'],
                    'same original input differs across comparisons')
        comparisons[name] = {'checks': len(checks), 'tensors': len(tensors),
                            'all_exact': all(r['exact_equal'] for r in tensors)}
    manifest=read(FORMAL/'source_manifest.json')
    for name,arm in [('native-admission01','native'),('control-admission01','control'),('optimized-audit004','optimized')]:
        folder=base/name;entry=read(folder/'entry.json');command=read(folder/'command.json')
        weighted=read(folder/'weighted.json');grad=read(folder/'gradnorm.json')
        require(command['seed']==contract['audit_seed'] and command['steps']==3 and command['audit'] is True and command['profile'] is False and command['exit_code']==0,'numerical audit invocation differs')
        require(command['stage34_arm']==arm and command['contract_sha256']==sha(root/'contract.json') and command['wrapper_sha256']==sha(root/'launch_weighted.py') and command['runner_sha256']==sha(root/'run_case.py'),'numerical audit controller differs')
        require(entry['native_sources_sha256']==manifest['files'] and entry['megatron_commit']==contract['megatron_commit'],'numerical audit native source identity differs')
        require(len(entry['models'])==1 and entry['models'][0]['parameters']==contract['model']['parameters'] and len(entry['models'][0]['parameter_names'])==P,'actual model parameter coverage differs')
        require(weighted['calls']==L*E*3 and weighted['sort_compat_calls']==weighted['router_calls']==len(weighted['expert_calls'])==L*3,'numerical expert/router dispatch differs')
        require(weighted['enabled'] is (arm=='optimized') and weighted['candidate_calls']==(L*E*3 if arm=='optimized' else 0) and weighted['native_calls']==(0 if arm=='optimized' else L*E*3) and weighted['fallback_calls']==0,'numerical candidate coverage differs')
        require(weighted['candidate_sha256']==(CANDIDATE_SHA if arm=='optimized' else None),'numerical candidate bytes differ')
        require(grad['calls']==(0 if arm=='native' else 3) and grad['enabled'] is (arm!='native') and grad['fallback_calls']==0,'numerical gradnorm coverage differs')
        require(grad['norm_audit']==[] if arm=='native' else len(grad['norm_audit'])==3 and all(row['bitwise_equal'] for row in grad['norm_audit']),'norm oracle differs')
    probe = read(base/('probe-'+version+'.json'))
    offset = read(base/('offset-'+version+'.json'))
    diagnostic = read(base/('diagnostics-'+version+'.json'))
    require(probe['status'] == 'passed' and len(probe['checks']) == 44 and len(probe['guards']) == 2
            and all(r['passed'] for r in probe['checks']+probe['guards']), 'independent probe failed')
    require(offset['passed'] is True and len(offset['checks']) == 2
            and all(r['pointer_alignment_mod16'] == [2, 2] and r['passed'] for r in offset['checks']),
            'real unaligned views not verified')
    require(diagnostic['status'] == 'complete' and diagnostic['passed'] is True and len(diagnostic['checks']) == L*3*E
            and all(all(z['exact_equal'] for z in r['native_replay'].values())
                    and all(z['within_original_tolerance'] for z in r['checks'].values())
                    and r['fallback_calls'] == 0 for r in diagnostic['checks']), '48 real replay cases missing')
    require(diagnostic['candidate_sha256'] == sha(candidate)
            and diagnostic['verifier_sha256'] == sha(root/'replay.py'), 'diagnostic provenance differs')
    for name in ('native-admission01','control-admission01','optimized-audit004'):
        weighted=read(base/name/'weighted.json')
        for rel,row in weighted['raw_files'].items():
            value=REMOTE+model+'/'+name+'/'+rel
            raw_rows[value]=dict(row,path=value)
    diagnostic_weighted=read(base/'native-admission01/weighted.json')
    for index, row in enumerate(diagnostic['checks'],1):
        require(row['index'] == index and row['capture_sha256'] ==
                diagnostic_weighted['raw_files']['activation/capture-%03d.pt'%index]['sha256'],
                'actual native capture identity differs')
    for r, filename, controller in [(probe, 'probe-'+version+'.json', 'probe.py'),
                                     (offset, 'offset-'+version+'.json', 'probe_offset.py')]:
        require(r['candidate_sha256'] == sha(candidate) and r['verifier_sha256'] == sha(root/controller),
                'probe identity differs')
    require(audit['probe_sha256'] == sha(base/('probe-'+version+'.json'))
            and audit['offset_probe_sha256'] == sha(base/('offset-'+version+'.json')), 'probe bytes differ')
    available, missing = [], []
    for value, row in raw_rows.items():
        local = local_original(value, base)
        if local is None:
            missing.append(value)
        else:
            require(local.stat().st_size == row['bytes'] and sha(local) == row['sha256'], 'original raw bytes differ')
            available.append(value)
    if require_raw:
        require(not missing, 'complete original raw bytes required locally')
    if audit_only:
        return {'status':'numerical_receipt_verified','model':model,'comparisons':comparisons,'candidate_sha256':sha(candidate),'available_original_inputs':len(available),'missing_original_inputs':missing,'original_byte_coverage_complete':not missing,'arithmetic_replay':False,'gpu_rerun':False}
    timing_base=base/'confirmation' if confirmation else base
    performance = read(timing_base/('performance-'+version+'.json'))
    require(performance['status'] == 'validated' and performance['seeds'] == contract['performance']['independent_seeds']
            and len(performance['pairs']) == 3 and len(performance['completed_cases']) == 9,
            'complete rotated nine-process timing required')
    require(performance['candidate_sha256'] == sha(candidate)
            and performance['contract_sha256'] == sha(root/'contract.json')
            and performance['runner_sha256'] == sha(root/'run_performance.py')
            and performance['audit_sha256'] == sha(base/('audit-'+version+'.json')), 'performance provenance differs')
    parser_path = FORMAL/'summarize_performance.py'
    require(sha(parser_path) == '6c2d1f7955f7e474a3de0b9e4edc156bc9ba90b5f23c7a5fce696ee736dcd2d3', 'native log parser differs')
    spec = importlib.util.spec_from_file_location('moe_archive_native_parser', parser_path)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    source_manifest = read(FORMAL/'source_manifest.json')
    loss_comparisons=[]
    for seed, order, pair in zip(performance['seeds'], ORDERS, performance['pairs']):
        require(pair['seed'] == seed and pair['order'] == order, 'timing seed/order differs')
        losses={}
        for arm in order:
            case = pair['cases'][arm]; folder = timing_base/case['case']
            for name, digest in case['raw_sha256'].items():
                require(sha(folder/name) == digest, 'original timing input differs')
            entry, command, weighted, grad = (read(folder/name) for name in
                ('entry.json','command.json','weighted.json','gradnorm.json'))
            require(entry['native_sources_sha256'] == source_manifest['files']
                    and entry['megatron_commit'] == contract['megatron_commit'], '600 native source identities differ')
            require(command['steps'] == 120 and command['seed'] == seed and command['stage34_arm'] == arm
                    and command['audit'] is command['profile'] is False and command['exit_code'] == 0,
                    'timing process differs')
            require(weighted['calls'] == W and weighted['sort_compat_calls'] == L*120
                    and weighted['router_calls'] == 0 and weighted['expert_calls'] == weighted['configurations'] == []
                    and weighted['fallback_calls'] == 0 and weighted['enabled'] is (arm == 'optimized')
                    and weighted['candidate_calls'] == (W if arm == 'optimized' else 0), 'weighted dispatch differs')
            require(grad['calls'] == (0 if arm == 'native' else 120) and grad['norm_audit'] == [],
                    'gradnorm instrumentation in timing')
            for name, digest in case['dispatch_sha256'].items():
                require(sha(folder/name) == digest, 'original dispatch bytes differ')
            original_log=(folder/'stdout.log').read_text(encoding='utf-8')
            intervals = parser.parse_intervals(original_log)
            losses[arm]=logged_losses(original_log)
            require(case['intervals'] == intervals, 'original intervals differ')
            close(case['mean_iteration_ms'], statistics.mean(r['mean_iteration_ms'] for r in intervals if r['retained']))
        for left, right in zip(order, order[1:]):
            require(datetime.fromisoformat(pair['cases'][left]['finished_utc']) <=
                    datetime.fromisoformat(pair['cases'][right]['started_utc']), 'timing processes overlap')
        loss_comparisons.append({'seed':seed,'printed_intervals':12,
            'iterations':list(range(10,121,10)),
            'native_control_equal':losses['native']==losses['control'],
            'native_optimized_equal':losses['native']==losses['optimized'],
            'scope':'Printed lm_loss and router_loss strings only; no120-step tensor/convergence assertion.'})
    for metric, left, right in [('vs_native','native','optimized'),('vs_control','control','optimized'),
                                ('control_vs_native','native','control')]:
        ratios = [r['cases'][left]['mean_iteration_ms']/r['cases'][right]['mean_iteration_ms'] for r in performance['pairs']]
        ratio = math.exp(statistics.mean(math.log(z) for z in ratios))
        close(performance[metric]['ratio'],ratio); close(performance[metric]['gain_percent'],100*(ratio-1))
        require(performance[metric]['all_seeds_faster'] is all(z>1 for z in ratios), 'all-seed promotion differs')
    promoted = performance['vs_control']['ratio'] > 1.01 and performance['vs_control']['all_seeds_faster']
    require(performance['promoted'] is promoted, 'fixed promotion rule differs')
    return {'status':'receipt_verified','version':version,'model':model,'confirmation':confirmation,'comparisons':comparisons,
        'vs_control':performance['vs_control'],'vs_native':performance['vs_native'],'promoted':promoted,
        'available_original_inputs':len(available),'missing_original_inputs':missing,
        'original_byte_coverage_complete':not missing,'verifier_sha256':sha(__file__),
        'loss_print_comparisons':loss_comparisons,'backend_receipt_sha256':sha(backend_path),
        'gpu_rerun':False,'arithmetic_replay':False,
        'scope':'Recorded fixed GPU gates and original logs verified on CPU; missing large raw files stay explicit in CI.'}



def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('long','experts'),required=True)
    p.add_argument('--base',type=Path);p.add_argument('--audit-only',action='store_true');p.add_argument('--require-raw',action='store_true');p.add_argument('--confirmation',action='store_true');a=p.parse_args()
    print(json.dumps(verify(a.model,a.base,a.require_raw,a.confirmation,a.audit_only),indent=2))


if __name__=='__main__':main()
