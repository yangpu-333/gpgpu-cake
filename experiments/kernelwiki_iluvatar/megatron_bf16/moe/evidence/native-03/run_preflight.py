"""Deploy only the tentative native MoE preflight and retain its raw logs/receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


HERE = Path(__file__).resolve().parent
REMOTE = '/tmp/kda-moe-preflight-stage33'
BF = '/tmp/kda-bf16-continuation/megatron_bf16'
NATIVE = '/tmp/kda-bf16-continuation/megatron-native-5be9626'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='bi-v150')
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--bool-sort-compat', action='store_true')
    a = parser.parse_args()
    assert a.suffix.replace('-', '').isalnum()
    evidence = HERE / 'evidence' / a.suffix
    assert not evidence.exists(), 'immutable preflight output already exists'
    evidence.mkdir(parents=True)
    (evidence / 'launch_preflight.py').write_bytes((HERE / 'launch_preflight.py').read_bytes())
    (evidence / 'run_preflight.py').write_bytes(Path(__file__).read_bytes())
    original = json.loads((HERE.parent / 'attention/contracts/main.json').read_text())
    argv = original['training_argv'] + ['--num-experts', '4', '--moe-router-topk', '2',
        '--expert-model-parallel-size', '1', '--moe-token-dispatcher-type', 'allgather',
        '--moe-router-load-balancing-type', 'aux_loss', '--moe-aux-loss-coeff', '0.01']
    contract = {'version': 'native-moe-tentative-preflight-v1', 'date': '2026-10-08',
        'megatron_commit': original['megatron_commit'], 'training_argv': argv,
        'bool_sort_compatibility': a.bool_sort_compat,
        'scope': 'Native MoE availability only; all dense candidate hooks disabled. '
                 'Proposed4L H512 FFN2048 S128 B2 V1024,4experts/topk2. '
                 'Actual parameter count and expert backend must be observed; not an optimization acceptance contract.'}
    data = (json.dumps(contract, indent=2) + '\n').encode()
    (evidence / 'contract.json').write_bytes(data)
    launcher = 'launch_preflight-' + a.suffix + '.py'
    payload = {launcher: (HERE / 'launch_preflight.py').read_bytes().decode(),
               a.suffix + '/contract.json': data.decode()}
    program = """import json
from pathlib import Path
root=Path(ROOT)
for name,content in PAYLOAD.items():
 p=root/name
 if p.exists():assert p.read_bytes()==content.encode(),str(p)
 else:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(content.encode())
print(json.dumps({'status':'deployed'}))
""".replace('ROOT', repr(REMOTE)).replace('PAYLOAD', repr(payload))
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', a.host, '/root/miniconda3/bin/python -'],
                            input=program.encode(), capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode(errors='replace')
    root = REMOTE + '/' + a.suffix
    command = ['/root/miniconda3/bin/python', '-u', REMOTE + '/' + launcher,
               '--moe-receipt', root + '/moe.json', '--delegate-launcher', BF + '/launch_training.py',
               '--megatron-root', NATIVE, '--contract', root + '/contract.json',
               '--entry-evidence', root + '/entry.json'] + (['--bool-sort-compat'] if a.bool_sort_compat else []) + argv + ['--train-iters', '3',
               '--seed', '33101', '--log-interval', '1']
    (evidence / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    env = ('env LD_LIBRARY_PATH=/usr/local/corex/lib64:/usr/local/openmpi/lib '
           'PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages '
           'MASTER_ADDR=127.0.0.1 MASTER_PORT=29533 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=0 ')
    (evidence / 'environment-command.txt').write_text(env)
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', a.host,
               env + ' '.join(shlex.quote(part) for part in command)], capture_output=True, timeout=180)
    (evidence / 'stdout.log').write_bytes(result.stdout)
    (evidence / 'stderr.log').write_bytes(result.stderr)
    fetch = """import json,base64
from pathlib import Path
root=Path(ROOT);values={}
for name in ('entry.json','moe.json'):
 p=root/name
 if p.is_file():values[name]=base64.b64encode(p.read_bytes()).decode()
print(json.dumps(values))
""".replace('ROOT', repr(root))
    pulled = subprocess.run(['ssh', '-o', 'BatchMode=yes', a.host, '/root/miniconda3/bin/python -'],
                            input=fetch.encode(), capture_output=True, timeout=60)
    assert pulled.returncode == 0, pulled.stderr.decode(errors='replace')
    import base64
    for name, content in json.loads(pulled.stdout).items():
        (evidence / name).write_bytes(base64.b64decode(content))
    receipt = {'status': 'completed' if result.returncode == 0 else 'failed',
               'exit_code': result.returncode, 'performance_claims': [],
               'wrapper_sha256': hashlib.sha256((HERE / 'launch_preflight.py').read_bytes()).hexdigest(),
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.iterdir() if p.is_file()}}
    (evidence / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    return 0 if result.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
