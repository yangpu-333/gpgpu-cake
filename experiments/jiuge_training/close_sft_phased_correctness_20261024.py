"""Close reports and preserve CPU-reviewed correctness; never launch training or Git."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import zipfile
from drive_sft_pinning_validation import SSH,SCP,PREP

BASE=Path(__file__).resolve().parent


def once(path,data):
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Preserve existing distinct result: '+str(path))
    else:
        with path.open('xb') as f:f.write(data)


def main():
    result=json.loads((BASE/'results/sft-phased-correctness-independent-review-20261024-03.json').read_text(encoding='utf-8'))
    driver=BASE/'results/sft-phased-correctness-driver-20261024-03'
    status=json.loads((driver/'status.json').read_text(encoding='utf-8'))
    remote=json.loads((driver/'remote-controller-latest.json').read_text(encoding='utf-8'))
    keys=json.loads((driver/'final-key-verification.json').read_text(encoding='utf-8'))
    assert status['status']==remote['status']=='completed'
    assert result['passed'] and result['full_state_and_native_resume_semantics_passed']
    assert not result['formal_performance_passed'] and not result['formal_performance_started']
    assert len(result['numeric_runs'])==10 and all(x['max_abs_error']==0 for x in result['numeric_runs'])
    bundle=driver/'final-key-results.zip'
    assert hashlib.sha256(bundle.read_bytes()).hexdigest()==result['key_bundle_sha256']==keys['bundle_sha256']
    assert keys['passed'] and keys['verified_key_bytes']<=64*1024**2
    for filename in ('sft-phased-acceptance-source-manifest-20261024-02.json','sft-phased-correctness-postprocess-source-20261024-03.json'):
        for n,h in json.loads((BASE/'results'/filename).read_text(encoding='utf-8'))['files'].items():
            assert hashlib.sha256((BASE/n).read_bytes()).hexdigest()==h,n
    with zipfile.ZipFile(bundle) as z:
        root='sft-phased-correctness-postprocess-20261024-03/'
        primary=json.loads(z.read(root+'primary-archive-verification.json'))
        semantic=json.loads(z.read(root+'semantics-verification.json'))
        checkpoint=json.loads(z.read(root+'checkpoint-sha256.json'))
        assert primary['passed'] and primary['primary_complete_raw_values_verified'] and primary['checkpoint_resume_tested']
        assert semantic['passed'] and semantic['sft_mask_gate'] and semantic['tp_actual_inputs_equal']
        once(BASE/'results/sft-phased-complete-primary-verification-20261024-03.json',z.read(root+'primary-archive-verification.json'))
        once(BASE/'results/sft-phased-complete-checkpoint-SHA-20261024-03.json',z.read(root+'checkpoint-sha256.json'))
    closure={'passed':True,'completed_utc':remote['updated_utc'],'full_state_and_native_resume_semantics_passed':True,
             'all10_numeric_runs_max_absolute_error':0,'ACT_raw_chunks_verified':result['ACT_raw_chunks_verified'],
             'checkpoint_files':len(checkpoint['files']),'checkpoint_bytes':sum(x['bytes'] for x in checkpoint['files']),
             'local_key_zip_bytes':bundle.stat().st_size,'local_key_files':keys['verified_key_files'],
             'local_key_uncompressed_bytes':keys['verified_key_bytes'],'key_bundle_sha256':keys['bundle_sha256'],
             'raw_tensors_downloaded':False,'formal_performance_started':False,'formal_performance_passed':False,
             'formal_blocker':'new explicitly authorized same-card time window required; original3h reserve expired',
             'all_new_double_GPU_jobs_closed':True,'double_GPU_cumulative_points':'0.00','old_V100_retained':True}
    once(BASE/'results/sft-phased-correctness-closure-20261024-03.json',(json.dumps(closure,indent=2)+'\n').encode())
    readme=BASE/'README.md';text=readme.read_text(encoding='utf-8')
    section=(f'\n### 23.5 正确性与原件校验闭合，正式性能待新窗口\n\n'
        f'三步全状态、原生iteration2检查点恢复及第3步、完整SFT语义已通过。'
        f'ACT逐块复核{result["ACT_raw_chunks_verified"]:,}个唯一原始数值块，覆盖10个数值运行；'
        f'全部受检字段最大绝对误差为0。检查点SHA清单覆盖{closure["checkpoint_files"]}份文件、'
        f'{closure["checkpoint_bytes"]:,} bytes。本机再次核对最终关键包及10份数值清单SHA，'
        f'见[最终独立复核](results/sft-phased-correctness-independent-review-20261024-03.json)与'
        f'[阶段闭合收据](results/sft-phased-correctness-closure-20261024-03.json)。\n\n'
        f'本机最终关键包{closure["local_key_zip_bytes"]:,} bytes，'
        f'{closure["local_key_files"]}份关键文件、解压{closure["local_key_uncompressed_bytes"]:,} bytes，'
        f'低于64MiB门槛；原始张量、检查点和完整日志均保留ACT。两个新双卡作业均已结束、累计0.00点，'
        f'旧V100保留。原编排误判记录与169份科学来源不覆盖。\n\n'
        f'**本阶段完成的是正确性、断点恢复和原件完整性，尚未获得新候选的正式吞吐结果。** '
        f'正式性能仍需另行明确授权、匹配同一C9/AF物理卡对的双卡窗口；'
        f'沿用三种子、三臂轮换、120步剔除20步、GPU同步`time.time`及慢rank、原CV复测、'
        f'相对增量baseline和control的全部配对种子及几何平均门槛。'
        f'历史CPT +84.4%、SFT pinned +92.9%与诊断筛选数字均不替代该结果。本轮收尾后暂停。\n')
    if '### 23.5 ' in text:
        assert text.endswith(section)
    else:readme.write_bytes((text+section).encode('utf-8'))
    brief=BASE/'BRIEF.md';text=brief.read_text(encoding='utf-8')
    paragraph=(f'**10月24日验收结论：** 新候选的三步全状态、原生检查点恢复及下一步、完整SFT语义均已通过，'
        f'全部受检字段最大误差为0。ACT已核验{result["ACT_raw_chunks_verified"]:,}个唯一原始数值块，'
        f'本机最终关键包及10份数值清单的独立复核通过。恢复初始化虚拟步的编排计数误判已通过版本化CPU复核纠正，'
        f'旧记录与科学来源不覆盖。**正式吞吐仍待另一个获授权的同卡对双卡窗口，本轮没有新增加速比。** '
        f'双卡已结束、累计0.00点，原始数据留ACT，本机只保留关键结果；本阶段收尾后暂停。'
        f'[完整证据与后续标准](README.md#235-正确性与原件校验闭合正式性能待新窗口)\n\n')
    text,n=re.subn(r'\*\*10月24日验收(?:进度|结论)：\*\*.*?\n\n',lambda _:paragraph,text,count=1,flags=re.S)
    assert n==1;brief.write_bytes(text.encode('utf-8'))
    state_path=BASE/'results/active-validation-state.json';state=json.loads(state_path.read_text(encoding='utf-8'));s=state['sft_phased_acceptance_20261024']
    s.update(status='correctness-complete-formal-requires-new-authorization',CPU_stage='completed',CPU_postprocess_active=False,
             full_checkpoint_resume_passed=True,full_raw_SHA_gate_passed=True,local_independent_review_passed=True,
             correctness_completed_utc=remote['updated_utc'],formal_performance_started=False,formal_performance_passed=False,
             raw_chunks_verified=result['ACT_raw_chunks_verified'],reports_complete=True,resource_closed=True,
             delivery_finalizer_executed=True,
             next_step_requires_new_resource_authorization=True,closure_receipt='results/sft-phased-correctness-closure-20261024-03.json')
    state_path.write_bytes((json.dumps(state,indent=2,ensure_ascii=False)+'\n').encode('utf-8'))
    names=['README.md','BRIEF.md',Path(__file__).name,'review_sft_phased_resume_policy_20261024.py',
           'finish_sft_phased_correctness_20261024.py','drive_sft_phased_correctness_20261024.py',
           'results/sft-phased-correctness-closure-20261024-03.json','results/sft-phased-correctness-independent-review-20261024-03.json',
           'results/sft-phased-complete-primary-verification-20261024-03.json','results/sft-phased-complete-checkpoint-SHA-20261024-03.json',
           'results/sft-phased-correctness-postprocess-source-20261024-03.json','results/sft-phased-acceptance-source-manifest-20261024-02.json',
           'results/sft-phased-resume-gate-platform-closure-20261024-02.json','results/sft-phased-resume-gate-stopped-0points-20261024-02.jpg',
           'results/sft-phased-correctness-driver-20261024-03/final-key-verification.json',
           'results/sft-phased-correctness-driver-20261024-03/independent-key-verification.json']
    manifest={n:{'bytes':(BASE/n).stat().st_size,'sha256':hashlib.sha256((BASE/n).read_bytes()).hexdigest()} for n in names}
    archive=BASE/'results/sft-phased-correctness-final-materials-20261024-03.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
            for n in names:z.write(BASE/n,n)
            z.writestr('manifest.json',json.dumps(manifest,indent=2))
    with zipfile.ZipFile(archive) as z:
        assert json.loads(z.read('manifest.json'))==manifest
        assert all(hashlib.sha256(z.read(n)).hexdigest()==row['sha256'] for n,row in manifest.items())
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    subprocess.run([*SCP,str(archive),'huids25@192.168.5.200:'+PREP+'/'+archive.name],capture_output=True,check=True,timeout=60)
    code='''from pathlib import Path
import json,zipfile,hashlib
p=Path('''+repr(PREP)+''');bundle=p/'''+repr(archive.name)+''';assert hashlib.sha256(bundle.read_bytes()).hexdigest()=='''+repr(digest)+'''
dest=p/'sft-phased-correctness-postprocess-20261024-03/delivery-final'
with zipfile.ZipFile(bundle) as z:
 m=json.loads(z.read('manifest.json'))
 if not dest.exists():
  dest.mkdir()
  for n,r in m.items():
   assert not Path(n).is_absolute() and '..' not in Path(n).parts
   data=z.read(n);assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256']
  z.extractall(dest)
 for n,r in m.items():assert hashlib.sha256((dest/n).read_bytes()).hexdigest()==r['sha256']
 r={'passed':True,'folder':str(dest),'files':len(m),'bytes':sum(x['bytes'] for x in m.values()),'bundle_sha256':hashlib.sha256(bundle.read_bytes()).hexdigest(),'formal_performance_passed':False,'raw_tensors_downloaded':False}
 if (dest/'ACT-verification.json').exists():assert json.loads((dest/'ACT-verification.json').read_text())==r
 else:
  with (dest/'ACT-verification.json').open('x') as f:json.dump(r,f,indent=2)
 print(json.dumps(r))
'''
    remote_result=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=60)
    once(BASE/'results/sft-phased-correctness-final-ACT-preservation-20261024-03.json',remote_result.stdout.encode())
    print(json.dumps(closure));print(remote_result.stdout)


if __name__=='__main__':main()
