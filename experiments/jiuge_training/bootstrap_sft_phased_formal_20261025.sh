#!/bin/bash
# New user-authorized zero-point formal campaign; six hours INCLUDING setup/retries.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-phased-formal-bootstrap-20261025-01"
install -d -m 755 "$logdir"
phase_host=$(hostname)
phase_start_file="$logdir/campaign-first-start.epoch"
# noclobber prevents any retry/container from resetting this campaign clock.
if ! test -f "$phase_start_file"; then
  (set -o noclobber; date +%s > "$phase_start_file") || true
fi
phase_start=$(cat "$phase_start_file")
[[ "$phase_start" =~ ^[0-9]+$ ]]
phase_left=$((phase_start + 21600 - $(date +%s)))
if test "$phase_left" -le 0; then exit 124; fi
if test "${JIUGE_FORMAL_GUARD:-}" != "$phase_host"; then
  exec timeout --signal=KILL "$phase_left" env JIUGE_FORMAL_GUARD="$phase_host" /bin/bash "$0"
fi
exec > >(tee -a "$logdir/bootstrap-$phase_host.log") 2>&1
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
gate="$prep/sft-phased-formal-approved-20261025-01.json"
while true; do
  if test -f "$gate"; then
    if "$python" - "$gate" "$phase_start" <<'PY'
import hashlib,json,socket,sys,time
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['hostname']==socket.gethostname()
assert r['max_charge_points']==0 and r['startup_limit_seconds']==21600
assert r['stop_before_epoch']==int(sys.argv[2])+21600
if time.time()>=r['stop_before_epoch']:raise SystemExit(124)
p=Path('/home/huids25/kda-jiuge-prep-20261014')
for name,sha in r['source_files_sha256'].items():
    assert '/' not in name and hashlib.sha256((p/name).read_bytes()).hexdigest()==sha,name
print('exact-new-formal-allocation-and-frozen-sources-approved',flush=True)
PY
    then break; else phase_code=$?; if test "$phase_code" -eq 124; then exit 124; fi; fi
  fi
  sleep 5
done
chown 1001:1001 "$logdir"
phase_left=$((phase_start + 21600 - $(date +%s)))
if test "$phase_left" -le 0; then exit 124; fi
set +e
timeout --signal=KILL "$phase_left" setpriv --reuid=1001 --regid=1001 --clear-groups \
 env -u PYTHONPATH -u PYTHONHOME HOME=/home/huids25 USER=huids25 LOGNAME=huids25 \
 /bin/bash -c '"$1" "$2/probe_sft_phased_formal_20261025.py" && exec "$1" -u "$2/continue_sft_phased_formal_20261025.py"' \
 phased-formal "$python" "$prep" > "$logdir/controller-$phase_host.stdout.log" 2> "$logdir/controller-$phase_host.stderr.log"
phase_code=$?
set -e
printf 'controller_exit_code=%s\n' "$phase_code"
date -u +%FT%TZ
exit "$phase_code"
