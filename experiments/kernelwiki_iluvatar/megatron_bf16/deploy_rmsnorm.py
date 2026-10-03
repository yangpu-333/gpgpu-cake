"""Deploy this experiment's explicitly enumerated scripts, no credentials."""
import json,subprocess,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files={'probe_rmsnorm_candidate.py':ROOT/'probe_rmsnorm_candidate.py','launch_rmsnorm.py':ROOT/'launch_rmsnorm.py',
       'candidate.py':ROOT/'rmsnorm_candidates/001/candidate.py',
       'run_rmsnorm_audit.py':ROOT/'run_rmsnorm_audit.py'}
code="import sys,json;from pathlib import Path;p=Path('/tmp/kda-rmsnorm-stage27');p.mkdir(exist_ok=True);d=json.load(sys.stdin);[(p/k).write_text(v) for k,v in d.items() if not (p/k).exists()]"
subprocess.run(['ssh','bi-v150',"/root/miniconda3/bin/python -c "+shlex.quote(code)],input=json.dumps({k:v.read_text() for k,v in files.items()}),text=True,check=True)
