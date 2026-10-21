"""Launch one immutable native 7B CPT case on the allocated two-GPU job."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PREP=Path('/home/huids25/kda-jiuge-prep-20261014')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--steps',type=int,default=3)
    parser.add_argument('--seq-length',type=int,default=128)
    parser.add_argument('--reference',action='store_true')
    parser.add_argument('--capture-windows',action='store_true')
    parser.add_argument('--capture-inputs',action='store_true')
    args=parser.parse_args()
    if not 2 <= args.seq_length <= 32768 or args.steps < 1:
        raise ValueError('Positive steps and sequence length within the real model limit required')
    if args.capture_inputs and (args.capture_windows or args.steps < 3):
        raise ValueError('Capture the first three inputs in a separate admission run')
    if not args.case.replace('-','').replace('_','').isalnum():
        raise ValueError('Simple immutable case name required')
    folder=PREP/'runs'/args.case
    folder.mkdir(parents=True,exist_ok=False)
    argv=['--tensor-model-parallel-size','2','--pipeline-model-parallel-size','1',
          '--num-layers','28','--hidden-size','3584','--ffn-hidden-size','18944',
          '--num-attention-heads','28','--num-query-groups','28',
          '--seq-length',str(args.seq_length),'--max-position-embeddings','32768',
          '--micro-batch-size','1','--global-batch-size','1','--train-iters',str(args.steps),
          '--lr','1e-5','--min-lr','1e-5','--lr-decay-style','constant','--lr-warmup-iters','0',
          '--optimizer','adam','--adam-beta1','0.9','--adam-beta2','0.95','--adam-eps','1e-8',
          '--weight-decay','0.1','--clip-grad','1.0',
          '--bf16','--accumulate-allreduce-grads-in-fp32','--use-distributed-optimizer',
          '--use-precision-aware-optimizer','--optimizer-cpu-offload','--optimizer-offload-fraction','1.0',
          '--use-torch-optimizer-for-cpu-offload','--overlap-cpu-optimizer-d2h-h2d',
          '--transformer-impl','local','--no-initialization',
          '--position-embedding-type','rope','--rotary-base','1000000','--rotary-percent','1.0','--no-rope-fusion',
          '--normalization','RMSNorm','--norm-epsilon','1e-6','--swiglu',
          '--disable-bias-linear','--add-qkv-bias','--untie-embeddings-and-output-weights',
          '--hidden-dropout','0','--attention-dropout','0','--make-vocab-size-divisible-by','1',
          '--vocab-size','73448','--no-persist-layer-norm','--no-gradient-accumulation-fusion',
          '--no-masked-softmax-fusion','--attention-softmax-in-fp32',
          '--recompute-granularity','full','--recompute-method','uniform','--recompute-num-layers','1',
          '--data-path',str(PREP/'datasets/alpaca-cpt-v1_text_document'),
          '--tokenizer-type','HuggingFaceTokenizer','--tokenizer-model',str(PREP/'checkpoints/9G7B_MHA'),
          '--trust-remote-code','--split','1,0,0','--num-workers','0','--seed','63100',
          '--log-interval','1','--eval-iters','0','--eval-interval','10000',
          '--distributed-timeout-minutes','10']
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             str(PREP/'native_entry.py'),'--evidence',str(folder/'rank-{rank}/entry.json'),
             '--jiuge-run-root',str(folder)]
    if args.reference:
        command+=['--reference-dir',str(PREP/'runs/official-reference-01')]
    if args.capture_inputs:
        command+=['--capture-inputs']
    if args.capture_windows:
        if args.steps < 35:
            raise ValueError('Two capture windows require at least 35 complete steps')
        command+=['--capture-windows']
    command+=argv
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='8',TORCHINDUCTOR_COMPILE_THREADS='2',CUDA_DEVICE_MAX_CONNECTIONS='1',
                       CUDA_VISIBLE_DEVICES='0,1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    environment['PATH']=os.pathsep.join((str(PREP/'.venv/bin'),str(Path(sys.base_prefix)/'bin'),environment.get('PATH','')))
    record={'schema':'jiuge7b-native-case-v1','case':args.case,'started_utc':datetime.now(timezone.utc).isoformat(),
            'command':command,'environment':{key:environment[key] for key in
            ('OMP_NUM_THREADS','TORCHINDUCTOR_COMPILE_THREADS','CUDA_DEVICE_MAX_CONNECTIONS','CUDA_VISIBLE_DEVICES')},
            'source_files_sha256':{name:hashlib.sha256((PREP/name).read_bytes()).hexdigest() for name in
            ('native_entry.py','jiuge7b_bridge.py','run_native_case.py')},'formal_performance_run':False}
    snapshots=folder/'source'
    snapshots.mkdir()
    for name in ('native_entry.py','jiuge7b_bridge.py','run_native_case.py','trace_hooks.py','optimizer_evidence.py','input_evidence.py'):
        if (PREP/name).is_file():
            shutil.copyfile(PREP/name,snapshots/name)
    # This survives SIGKILL/OOM even if the process cannot write its final receipt.
    (folder/'case-start.json').write_text(json.dumps(record,indent=2)+'\n')
    with (PREP/'two-gpu-experiment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        before=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True,check=True)
        if before.stdout.strip():
            raise RuntimeError('Allocated GPU has another compute process; refusing to launch')
        record['resource_gate_before']=before.stdout
        with (folder/'stdout.log').open('x') as stdout,(folder/'stderr.log').open('x') as stderr:
            record['exit_code']=subprocess.run(command,cwd=PREP/'megatron-native-5be9626',env=environment,
                                               stdout=stdout,stderr=stderr,timeout=7200).returncode
        record['resource_gate_after']=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout
    record['finished_utc']=datetime.now(timezone.utc).isoformat()
    (folder/'case.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'case':args.case,'exit_code':record['exit_code'],'directory':str(folder)}),flush=True)
    return record['exit_code']


if __name__=='__main__':
    raise SystemExit(main())
