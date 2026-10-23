#!/bin/bash
# At most3h including setup and waiting. No new login identity; existing UID only.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-phased-moments-bootstrap-20261023-03"
install -d -m 755 "$logdir"
moment_host=$(hostname)
moment_job=${moment_host%%-default*}
moment_start_file="$logdir/first-start-$moment_job.epoch"
if ! test -f "$moment_start_file"; then date +%s > "$moment_start_file"; fi
moment_campaign_start="$prep/sft-direct-zero-bootstrap-20261023-01/campaign-first-start.epoch"
if ! test -f "$moment_campaign_start"; then (set -C; cat "$moment_start_file" > "$moment_campaign_start") 2>/dev/null || test -f "$moment_campaign_start"; fi
moment_start=$(cat "$moment_campaign_start")
[[ "$moment_start" =~ ^[0-9]+$ ]]
moment_left=$((moment_start + 10800 - $(date +%s)))
if test "$moment_left" -le 0; then exit 124; fi
if test "${JIUGE_MOMENT_CAP_GUARD:-}" != "$moment_host"; then
  exec timeout --signal=KILL "$moment_left" env JIUGE_MOMENT_CAP_GUARD="$moment_host" /bin/bash "$0"
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
setpriv --reuid=1001 --regid=1001 --clear-groups env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 "$python" "$prep/probe_sft_direct_zero_allocation_20261023.py"
gate="$prep/sft-phased-moments-approved-20261023-03.json"
while true; do
  if test -f "$gate"; then
    if "$python" - "$gate" "$moment_start" <<'PY'
import hashlib,json,socket,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['hostname']==socket.gethostname()
assert r['max_charge_points']==0 and r['startup_limit_seconds']==10800
assert r['stop_before_epoch']<=int(sys.argv[2])+10800
if time.time()>=r['stop_before_epoch']:raise SystemExit(124)
p=Path('/home/huids25/kda-jiuge-prep-20261014')
for name,sha in r['source_files_sha256'].items():
    assert '/' not in name and hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print('exact-job-and-frozen-source-gate-passed',flush=True)
PY
    then break; else moment_code=$?; if test "$moment_code" -eq 124; then exit 124; fi; fi
  fi
  sleep 5
done
chown 1001:1001 "$logdir"
moment_approved_stop=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["stop_before_epoch"]))' "$gate")
moment_seconds=$((moment_approved_stop - $(date +%s)))
if test "$moment_seconds" -le 0; then exit 124; fi
set +e
timeout --signal=KILL "$moment_seconds" setpriv --reuid=1001 --regid=1001 --clear-groups \
  env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
  /bin/bash -c '"$1" "$2/probe_sft_runtime_phased_20261023.py" && "$1" "$2/admit_sft_first_step_phased_20261023.py" && exec "$1" -u "$2/continue_sft_screen_phased_20261023.py"' \
  moment-runtime "$python" "$prep" \
  > "$logdir/controller.stdout.log" 2> "$logdir/controller.stderr.log"
moment_code=$?
set -e
printf 'controller_exit_code=%s\n' "$moment_code"
date -u +%FT%TZ
exit "$moment_code"
