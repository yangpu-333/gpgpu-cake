#!/bin/bash
# Three-hour cap includes setup and idle gate time; no login identity is created.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-numa-bootstrap-20261021-01"
install -d -m 755 "$logdir"
numa_host=$(hostname)
numa_job=${numa_host%%-default*}
numa_start_file="$logdir/first-start-$numa_job.epoch"
if ! test -f "$numa_start_file"; then date +%s > "$numa_start_file"; fi
numa_start=$(cat "$numa_start_file")
[[ "$numa_start" =~ ^[0-9]+$ ]]
numa_remaining=$((numa_start + 10800 - $(date +%s)))
if test "$numa_remaining" -le 0; then exit 124; fi
if test "${JIUGE_SFT_NUMA_CAP_GUARD:-}" != "$numa_host"; then
  exec timeout --signal=KILL "$numa_remaining" env JIUGE_SFT_NUMA_CAP_GUARD="$numa_host" /bin/bash "$0"
fi
exec > >(tee -a "$logdir/bootstrap.log") 2>&1
date -u +%FT%TZ
hostname
export DEBIAN_FRONTEND=noninteractive
timeout --signal=KILL 600 apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 update
timeout --signal=KILL 600 apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 install -y --no-install-recommends \
  build-essential libgomp1 libstdc++6 libnuma1 procps util-linux ca-certificates \
  libexpat1 libffi8 liblzma5 libbz2-1.0 zlib1g libsqlite3-0 libgdbm6 libreadline8 libncursesw6
unset PYTHONPATH PYTHONHOME
python="$prep/.venv/bin/python"
gate="$prep/sft-numa-approved-20261021.json"
while true; do
  if test -f "$gate"; then
    if "$python" - "$gate" <<'PY'
import hashlib,json,socket,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['hostname']==socket.gethostname()
assert r['max_charge_points']==0 and r['startup_limit_seconds']==10800
if time.time()>=r['stop_before_epoch']: raise SystemExit(124)
p=Path('/home/huids25/kda-jiuge-prep-20261014')
for name,sha in r['source_files_sha256'].items():
    assert '/' not in name and hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print('exact-job-and-frozen-source-gate-passed',flush=True)
PY
    then break; else numa_code=$?; if test "$numa_code" -eq 124; then exit 124; fi; fi
  fi
  sleep 5
done
chown 1001:1001 "$logdir"
numa_seconds=$("$python" - "$gate" <<'PY'
import json,sys,time
from pathlib import Path
print(max(1,int(json.loads(Path(sys.argv[1]).read_text())['stop_before_epoch']-time.time())))
PY
)
set +e
timeout --signal=KILL "$numa_seconds" setpriv --reuid=1001 --regid=1001 --clear-groups \
  env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
  "$python" -u "$prep/continue_sft_numa_20261021.py" \
  > "$logdir/controller.stdout.log" 2> "$logdir/controller.stderr.log"
numa_code=$?
set -e
printf 'controller_exit_code=%s\n' "$numa_code"
date -u +%FT%TZ
exit "$numa_code"
