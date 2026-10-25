"""Close reports/ACT delivery only after independent formal review and platform stop."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
from drive_sft_pinning_validation import PREP, SSH, SCP

BASE=Path(__file__).resolve().parent
RESULTS=BASE/'results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=read(RESULTS/'sft-phased-formal-independent-review-20261025-01.json')
    driver=RESULTS/'sft-phased-formal-driver-20261025-01'
    status=read(driver/'status.json')
    platform=read(RESULTS/'sft-phased-formal-platform-closure-20261025-01.json')
    assert review['passed'] and status['status']=='completed'
    assert platform['job']=='sg-huids25-261025-ec72b'
    assert platform['resource_closed'] and platform['cumulative_points']=='0.00'
    assert platform['old_V100_retained']
    manifest=read(RESULTS/'sft-phased-formal-source-manifest-20261025-01.json')['files']
    for name,digest in manifest.items():assert sha(BASE/name)==digest,name
    supplemental=read(RESULTS/'sft-phased-formal-postprocess-sources-20261025-01.json')['files']
    for name,digest in supplemental.items():assert sha(BASE/name)==digest,name
    key=read(driver/'final-key-verification.json')
    assert key['passed'] and sha(driver/'final-key-results.zip')==key['bundle_sha256']==review['key_bundle_sha256']
    passed=review['formal_performance_passed']
    lines=['### 4.3 新moment候选正式验收（10月25日）','',
           '复用10月24日已闭合的候选自身全状态、原生断点恢复、SFT语义及18,601原始数值块SHA证据；169份科学源码未改。本轮在相同C9/AF双A100卡对重新独立计时，不复用旧性能数据。','',
           '| 轮次 | pinned基线 秒/步 | 同绑核control 秒/步 | 本地moment候选 秒/步 | 相对基线 | 相对control |',
           '|---|---:|---:|---:|---:|---:|']
    for item in review['rounds']:
        means={arm:sum(row['mean_seconds_per_step'] for row in item['cases'] if row['arm']==arm)/3
               for arm in ('native','control','optimized')}
        lines.append('| '+('初测' if item['name']=='primary' else '完整复测')+
          f" | {means['native']:.4f} | {means['control']:.4f} | {means['optimized']:.4f} | {(item['geomeans']['native']-1)*100:+.2f}% | {(item['geomeans']['control']-1)*100:+.2f}% |")
    lines+=['', '**'+('候选通过正式性能验收。' if passed else '候选未通过预定性能提升门槛，不推广为新增优化。')+'**', '',
            '各轮均为三种子、三臂轮换、120步剔除20步；沿用GPU同步time.time及两rank对齐窗口慢端，无profiler或数值审计/检查点IO。各百分比为配对种子吞吐比的几何平均减1；baseline/control种子CV达到10%时执行完整九案例复测。推广要求相对两条基线的每个种子均更快且几何平均均超过1%，必要两轮均通过。120批样本/mask及13条日志loss经独立复核。', '',
            '上述收益属于本地moment候选相对已验收pinned版本的增量；历史CPT +84.4%、SFT pinned +92.9%及筛选约9.2%保持独立。双卡已结束，累计0.00点，旧V100保留。原件留ACT，本机只保留关键结果；本阶段暂停。']
    section='\n'.join(lines)+'\n'
    brief=BASE/'BRIEF.md'; text=brief.read_text(encoding='utf-8')
    assert '### 4.3 新moment候选正式验收（10月25日）' not in text
    brief.write_text(text.rstrip()+'\n\n'+section,encoding='utf-8')
    report=BASE/'README.md'; text=report.read_text(encoding='utf-8')
    assert '## 24. 新moment候选正式性能验收（2026-10-25）' not in text
    report.write_text(text.rstrip()+'\n\n## 24. 新moment候选正式性能验收（2026-10-25）\n\n'+section+
      '\n独立复核：`results/sft-phased-formal-independent-review-20261025-01.json`。平台收尾：`results/sft-phased-formal-platform-closure-20261025-01.json`。\n',encoding='utf-8')
    closure={'passed':True,'formal_performance_passed':passed,'rounds':len(review['rounds']),
       'numeric_correctness_previously_closed':True,'new_scientific_sources':False,
       'key_bundle_sha256':key['bundle_sha256'],'raw_tensors_downloaded':False,
       'resource_closed':True,'cumulative_points':'0.00','old_V100_retained':True,
       'phase_paused':True,'utc':datetime.now(timezone.utc).isoformat()}
    output=RESULTS/'sft-phased-formal-closure-20261025-01.json'
    with output.open('x',encoding='utf-8') as f:json.dump(closure,f,indent=2)
    state_path=RESULTS/'active-validation-state.json';state=read(state_path)
    state['sft_phased_formal_20261025'].update(status='completed-paused',formal_performance_started=True,
       formal_performance_passed=passed,independent_review_passed=True,resource_closed=True,
       any_active_GPU_experiment=False,reports_complete=True,key_bundle_sha256=key['bundle_sha256'])
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    files=[BASE/'README.md',BASE/'BRIEF.md',output,RESULTS/'active-validation-state.json',
      RESULTS/'sft-phased-formal-independent-review-20261025-01.json',
      RESULTS/'sft-phased-formal-source-manifest-20261025-01.json',
      RESULTS/'sft-phased-formal-postprocess-sources-20261025-01.json',
      RESULTS/'sft-phased-formal-platform-closure-20261025-01.json',
      RESULTS/'sft-phased-formal-authorization-20261025-01.json',
      driver/'final-key-results.zip',driver/'final-key-verification.json',driver/'status.json']
    files.extend(BASE/name for name in manifest if '20261025' in name)
    files.extend(BASE/name for name in supplemental)
    files.append(Path(__file__).resolve())
    bundle=RESULTS/'sft-phased-formal-final-delivery-20261025-01.zip';catalog={}
    with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path in files:
            name=path.relative_to(BASE).as_posix()
            assert name not in catalog
            catalog[name]={'bytes':path.stat().st_size,'sha256':sha(path)}
            archive.writestr(name,path.read_bytes())
        archive.writestr('delivery-manifest.json',json.dumps(catalog,indent=2))
    remote=PREP+'/sft-phased-formal-20261025-01/delivery-final'
    code='from pathlib import Path; Path('+repr(remote)+').mkdir(exist_ok=False)'
    subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,check=True,timeout=60)
    subprocess.run([*SCP,str(bundle),'huids25@192.168.5.200:'+remote+'/delivery.zip'],check=True,timeout=120)
    code='''from pathlib import Path
import hashlib,json,zipfile
p=Path('''+repr(remote)+''');z=p/'delivery.zip'
assert hashlib.sha256(z.read_bytes()).hexdigest()=='''+repr(sha(bundle))+'''
with zipfile.ZipFile(z) as a:
    m=json.loads(a.read('delivery-manifest.json'))
    for n,r in m.items():
        assert not Path(n).is_absolute() and '..' not in Path(n).parts
        b=a.read(n);assert len(b)==r['bytes'] and hashlib.sha256(b).hexdigest()==r['sha256']
    a.extractall(p)
r={'passed':True,'files':len(m),'bytes':sum(v['bytes'] for v in m.values()),'sha256':hashlib.sha256(z.read_bytes()).hexdigest(),'ACT_directory':str(p)}
with (p/'ACT-verification.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
'''
    checked=subprocess.run([*SSH,PREP+'/.venv/bin/python -'],input=code,text=True,capture_output=True,check=True,timeout=120)
    with (RESULTS/'sft-phased-formal-final-ACT-preservation-20261025-01.json').open('x',encoding='utf-8') as f:f.write(checked.stdout)
    print(checked.stdout)


if __name__=='__main__':main()
