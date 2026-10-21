#!/bin/bash
# Container-only bootstrap for the authorized six-hour, two-A100 tracing job.
# No host settings, persistent credentials, model math, or optimizer are changed.
set -euo pipefail
prep=/home/huids25/kda-jiuge-prep-20261014
logdir="$prep/sft-ubuntu-bootstrap-20261017-01"
install -d -m 755 "$logdir" /run/sshd
exec > >(tee -a "$logdir/bootstrap.log") 2>&1
date -u +%FT%TZ
hostname
export DEBIAN_FRONTEND=noninteractive
export JIUGE_BASE_IMAGE=library/ubuntu:22.04
timeout --signal=KILL 900 apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update
timeout --signal=KILL 900 apt-get -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
  openssh-server build-essential git libgomp1 libstdc++6 libnuma1 numactl procps ca-certificates \
  libexpat1 libffi8 liblzma5 libbz2-1.0 zlib1g libsqlite3-0 libgdbm6 libreadline8 libncursesw6
getent group huids25 >/dev/null || groupadd -g 1001 huids25
id huids25 >/dev/null 2>&1 || useradd -u 1001 -g 1001 -M -d /home/huids25 -s /bin/bash huids25
test "$(id -u huids25)" = 1001
test "$(id -g huids25)" = 1001
ssh-keygen -A
/usr/sbin/sshd -t
/usr/sbin/sshd -e -p 22 -o AllowUsers=huids25 -o PasswordAuthentication=no \
  -o KbdInteractiveAuthentication=no -o PermitRootLogin=no -o PubkeyAuthentication=yes
# Public identity only. The private host keys remain in this container's /etc/ssh.
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub > "$logdir/host-ed25519-fingerprint.txt"
cat /etc/ssh/ssh_host_ed25519_key.pub > "$logdir/host-ed25519-public.txt"
unset PYTHONPATH PYTHONHOME
python="$prep/.venv/bin/python"
# The exact job identity and frozen sources are written through the already
# trusted V100/shared filesystem after the website returns the new job ID.
gate="$prep/sft-ubuntu-launch-approved-20261017.json"
while true; do
  if test -f "$gate"; then
    if "$python" - "$gate" <<'PY'
from pathlib import Path
import hashlib,json,socket,sys
record=json.loads(Path(sys.argv[1]).read_text())
assert socket.gethostname()==record['hostname']
assert record['max_charge_points']==0 and record['startup_limit_seconds']==21600
prep=Path('/home/huids25/kda-jiuge-prep-20261014')
for name,expected in record['source_files_sha256'].items():
    assert '/' not in name and (prep/name).is_file()
    assert hashlib.sha256((prep/name).read_bytes()).hexdigest()==expected,name
print(json.dumps({'launch_gate_passed':True,'hostname':socket.gethostname()}),flush=True)
PY
    then
      break
    fi
  fi
  sleep 5
done
date -u +%FT%TZ
set +e
runuser -u huids25 -- env -u PYTHONPATH -u PYTHONHOME \
  "$python" -u "$prep/continue_sft_tracing.py" \
  > "$logdir/controller.stdout.log" 2> "$logdir/controller.stderr.log"
code=$?
set -e
printf 'controller_exit_code=%s\n' "$code"
date -u +%FT%TZ
# Returning ends this allocation. The external timeout is an independent cap.
exit "$code"
