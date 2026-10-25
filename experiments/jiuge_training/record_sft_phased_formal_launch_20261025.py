"""Persist the human authorization and one bounded new formal campaign."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
if __name__=='__main__':
    driver=BASE/'results/sft-phased-formal-driver-20261025-01'
    driver.mkdir(exist_ok=False)
    start=1792892990
    budget={'job':'sg-huids25-261025-ec72b','hostname':'sg-huids25-261025-ec72b-default0-0',
            'user_explicit_authorization':'继续，授权新一轮双A100正式性能验收，费用上限0点，统一标准不变。先检查有没有其他人在用卡，原始数据留ACT，本机只保留关键结果，完成后停止双卡并暂停。',
            'max_charge_points':0,'GPU':2,'CPU':16,'memory_GiB':300,
            'campaign_start_epoch':start,'stop_before_epoch':start+21600,
            'stop_before_Beijing':datetime.fromtimestamp(start+21600,timezone(timedelta(hours=8))).isoformat(),
            'startup_limit_seconds':21600,'all_retries_share_deadline':True,
            'free_minutes_observed':720,'complete_early_and_exit':True,'keep_old_V100':True,
            'no_new_identity':True,'root_dependencies_only_training_UID_GID':[1001,1001],
            'originals':'ACT Ceph','local':'64MiB bounded keys only'}
    with (BASE/'results/sft-phased-formal-authorization-20261025-01.json').open('x',encoding='utf-8') as f:
        json.dump(budget,f,ensure_ascii=False,indent=2)
    state_path=BASE/'results/active-validation-state.json'
    state=json.loads(state_path.read_text(encoding='utf-8'))
    state['sft_phased_formal_20261025']={**budget,'status':'bootstrap-waiting-for-allocation-gate',
          'driver':'results/sft-phased-formal-driver-20261025-01/status.json',
          'ACT_controller':'/home/huids25/kda-jiuge-prep-20261014/sft-phased-formal-20261025-01/controller.json',
          'source_manifest':'results/sft-phased-formal-source-manifest-20261025-01.json',
          'source_files':179,'old169_unchanged':True,'formal_performance_started':False,
          'formal_performance_passed':False,'correctness_already_closed':True,
          'allocation_GPU_process_gate_pending':True,
          'platform_preflight':'inspur-gpu-04 four A100 unallocated, no active user Pods',
          'updated_utc':datetime.now(timezone.utc).isoformat()}
    state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'job':budget['job'],'stop_before_Beijing':budget['stop_before_Beijing']}))
