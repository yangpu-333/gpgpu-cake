#!/bin/bash
# Proposed separate free allocation: max11h including dependencies and approval.
# No new account, password, key or SSH service. Exit early when the controller ends.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-phased-acceptance-bootstrap-20261024-01"
install -d -m 755 "$logdir"
phase_host=$(hostname)
phase_campaign_start="$logdir/campaign-first-start.epoch"
if ! test -f "$phase_campaign_start"; then (set -C; date +%s > "$phase_campaign_start") 2>/dev/null || test -f "$phase_campaign_start"; fi
phase_start=$(cat "$phase_campaign_start")
[[ "$phase_start" =~ ^[0-9]+$ ]]
phase_left=$((phase_start + 39600 - $(date +%s)))
if test "$phase_left" -le 0; then exit 124; fi
if test "${JIUGE_PHASE_ACCEPT_GUARD:-}" != "$phase_host"; then
  exec timeout --signal=KILL "$phase_left" env JIUGE_PHASE_ACCEPT_GUARD="$phase_host" /bin/bash "$0"
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
setpriv --reuid=1001 --regid=1001 --clear-groups env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
  "$python" "$prep/probe_sft_direct_zero_allocation_20261023.py"
gate="$prep/sft-phased-acceptance-approved-20261024-01.json"
while true; do
  if test -f "$gate"; then
    if "$python" - "$gate" "$phase_start" <<'PY'
import hashlib,json,socket,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['hostname']==socket.gethostname()
assert r['max_charge_points']==0 and r['startup_limit_seconds']==39600
assert r['stop_before_epoch']<=int(sys.argv[2])+39600
if time.time()>=r['stop_before_epoch']:raise SystemExit(124)
p=Path('/home/huids25/kda-jiuge-prep-20261014')
for name,sha in r['source_files_sha256'].items():
    assert '/' not in name and hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print('exact-allocation-and-frozen-full-acceptance-gate-passed',flush=True)
PY
    then break; else phase_code=$?; if test "$phase_code" -eq 124; then exit 124; fi; fi
  fi
  sleep 5
done
chown 1001:1001 "$logdir"
phase_stop=$("$python" -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["stop_before_epoch"]))' "$gate")
phase_left=$((phase_stop - $(date +%s)))
if test "$phase_left" -le 0; then exit 124; fi
set +e
timeout --signal=KILL "$phase_left" setpriv --reuid=1001 --regid=1001 --clear-groups \
  env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
  /bin/bash -c '"$1" "$2/probe_sft_phased_acceptance_20261024.py" && exec "$1" -u "$2/continue_sft_phased_acceptance_20261024.py"' \
  phased-acceptance "$python" "$prep" \
  > "$logdir/controller.stdout.log" 2> "$logdir/controller.stderr.log"
phase_code=$?
set -e
printf 'controller_exit_code=%s\n' "$phase_code"
date -u +%FT%TZ
exit "$phase_code"
