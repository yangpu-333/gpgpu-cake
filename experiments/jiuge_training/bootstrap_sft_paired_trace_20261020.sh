#!/bin/bash
# Three-hour independent cap; shared-storage gate, no new login identity.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-paired-trace-bootstrap-20261020-01"
install -d -m 755 "$logdir"
trace_host=$(hostname)
trace_job=${trace_host%%-default*}
trace_start_file="$logdir/first-start-$trace_job.epoch"
if ! test -f "$trace_start_file"; then date +%s > "$trace_start_file"; fi
trace_start=$(cat "$trace_start_file")
[[ "$trace_start" =~ ^[0-9]+$ ]]
trace_remaining=$((trace_start + 10800 - $(date +%s)))
if test "$trace_remaining" -le 0; then exit 124; fi
if test "${JIUGE_PAIRED_TRACE_CAP_GUARD:-}" != "$trace_host"; then
  exec timeout --signal=KILL "$trace_remaining" env JIUGE_PAIRED_TRACE_CAP_GUARD="$trace_host" /bin/bash "$0"
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
gate="$prep/sft-paired-trace-approved-20261020.json"
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
    then break; else trace_code=$?; if test "$trace_code" -eq 124; then exit 124; fi; fi
  fi
  sleep 5
done
chown 1001:1001 "$logdir"
trace_seconds=$("$python" - "$gate" <<'PY'
import json,sys,time
from pathlib import Path
print(max(1,int(json.loads(Path(sys.argv[1]).read_text())['stop_before_epoch']-time.time())))
PY
)
set +e
timeout --signal=KILL "$trace_seconds" setpriv --reuid=1001 --regid=1001 --clear-groups \
  env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
  "$python" -u "$prep/continue_sft_paired_trace_20261020.py" \
  > "$logdir/controller.stdout.log" 2> "$logdir/controller.stderr.log"
trace_code=$?
set -e
printf 'controller_exit_code=%s\n' "$trace_code"
date -u +%FT%TZ
exit "$trace_code"
