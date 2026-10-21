#!/bin/bash
# Container-only bootstrap. No login identity, password, keys, or SSH service.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-pinning-formal-bootstrap-20261020-01"
install -d -m 755 "$logdir"
sft_host=$(hostname)
sft_job=${sft_host%%-default*}
sft_first_start_file="$logdir/first-start-$sft_job.epoch"
if ! test -f "$sft_first_start_file"; then
  date +%s > "$sft_first_start_file"
fi
sft_first_start=$(cat "$sft_first_start_file")
if ! [[ "$sft_first_start" =~ ^[0-9]+$ ]]; then
  echo 'invalid-persistent-start-budget' >&2
  exit 2
fi
sft_remaining=$((sft_first_start + 21600 - $(date +%s)))
if test "$sft_remaining" -le 0; then
  exit 124
fi
if test "${JIUGE_SFT_FORMAL_CAP_GUARD:-}" != "$sft_host"; then
  exec timeout --signal=KILL "$sft_remaining" env JIUGE_SFT_FORMAL_CAP_GUARD="$sft_host" /bin/bash "$0"
fi
exec > >(tee -a "$logdir/bootstrap.log") 2>&1
date -u +%FT%TZ
hostname
export DEBIAN_FRONTEND=noninteractive
timeout --signal=KILL 900 apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 update
timeout --signal=KILL 900 apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 install -y --no-install-recommends \
  build-essential libgomp1 libstdc++6 libnuma1 procps util-linux ca-certificates \
  libexpat1 libffi8 liblzma5 libbz2-1.0 zlib1g libsqlite3-0 libgdbm6 libreadline8 libncursesw6
unset PYTHONPATH PYTHONHOME
python="$prep/.venv/bin/python"
gate="$prep/sft-pinning-formal-approved-20261020.json"
while true; do
  if test -f "$gate"; then
    if "$python" - "$gate" <<'PY'
import hashlib,json,socket,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['hostname']==socket.gethostname()
assert r['max_charge_points']==0 and r['startup_limit_seconds']==21600
if time.time()>=r['stop_before_epoch']:
    raise SystemExit(124)
p=Path('/home/huids25/kda-jiuge-prep-20261014')
for name,sha in r['source_files_sha256'].items():
    assert '/' not in name and hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print('exact-job-and-source-gate-passed',flush=True)
PY
    then
      break
    else
      sft_gate_code=$?
      if test "$sft_gate_code" -eq 124; then exit 124; fi
    fi
  fi
  sleep 5
done
chown 1001:1001 "$logdir"
seconds=$("$python" - "$gate" <<'PY'
import json,sys,time
from pathlib import Path
print(max(1,int(json.loads(Path(sys.argv[1]).read_text())['stop_before_epoch']-time.time())))
PY
)
set +e
timeout --signal=KILL "$seconds" setpriv --reuid=1001 --regid=1001 --clear-groups \
  env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
  "$python" -u "$prep/continue_sft_pinning_formal_20261020.py" \
  > "$logdir/controller.stdout.log" 2> "$logdir/controller.stderr.log"
code=$?
set -e
printf 'controller_exit_code=%s\n' "$code"
date -u +%FT%TZ
exit "$code"
