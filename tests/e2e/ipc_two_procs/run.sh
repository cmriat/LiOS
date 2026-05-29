#!/usr/bin/env bash
set -euo pipefail

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Create a temp work dir with named pipes for clean bidirectional IO
workdir=$(mktemp -d -t ipc2p-XXXXXX)
cleanup() {
  set +e
  [[ -p "$workdir/w_in" ]] && rm -f "$workdir/w_in"
  [[ -p "$workdir/w_out" ]] && rm -f "$workdir/w_out"
  [[ -p "$workdir/r_in" ]] && rm -f "$workdir/r_in"
  [[ -p "$workdir/r_out" ]] && rm -f "$workdir/r_out"
  [[ -f "$workdir" ]] && rm -rf "$workdir" || true
}
trap cleanup EXIT

mkfifo "$workdir/w_in" "$workdir/w_out" "$workdir/r_in" "$workdir/r_out"

# Start writer (independent Python via pixi)
(
  set -eo pipefail
  pixi run python "$dir/writer.py" <"$workdir/w_in" >"$workdir/w_out"
) &
w_pid=$!

# Open fds for interactive control
exec 3>"$workdir/w_in"
exec 4<"$workdir/w_out"

read_line_timeout() {
  local fd="$1"; shift
  local timeout_sec="${1:-20}"; shift || true
  local line
  # bash read with -t timeout; returns non-zero on timeout
  if IFS= read -r -t "$timeout_sec" -u "$fd" line; then
    printf '%s' "$line"
    return 0
  fi
  return 1
}

read_until_prefix() {
  local fd="$1"; local prefix="$2"; local timeout_sec="${3:-20}"
  local start_ts=$(date +%s)
  local line
  while true; do
    if ! line=$(read_line_timeout "$fd" 1); then
      local now=$(date +%s)
      if (( now - start_ts >= timeout_sec )); then
        echo "Timeout waiting for $prefix" >&2
        return 1
      fi
      continue
    fi
    echo "$line"
    [[ "$line" == $prefix* ]] && return 0
  done
}

# Parse writer readiness and payload
ready_line=$(read_until_prefix 4 "READY" 30)
b64_line=$(read_until_prefix 4 "B64" 30)
sem_line=$(read_until_prefix 4 "SEM" 30)
payload_b64=${b64_line#B64 }

# Start reader with the received payload
(
  set -eo pipefail
  pixi run python "$dir/reader.py" --payload "$payload_b64" <"$workdir/r_in" >"$workdir/r_out"
) &
r_pid=$!

exec 5>"$workdir/r_in"
exec 6<"$workdir/r_out"

r_ready_line=$(read_until_prefix 6 "READY" 30)

# Determine CUDA availability on both ends
if [[ "$ready_line" == *"cuda=1"* && "$r_ready_line" == *"cuda=1"* ]]; then
  both_cuda=1
else
  both_cuda=0
fi

# Phase 1: writer writes v1, reader reads
v1=1.2345
echo "WRITE $v1" >&3
w_wrote_line=$(read_until_prefix 4 "WROTE" 10)

echo "READMEAN" >&5
r_mean1_line=$(read_until_prefix 6 "MEAN" 10)

wm1=${w_wrote_line#WROTE }
rm1=${r_mean1_line#MEAN }

if (( both_cuda == 1 )); then
  awk -v a="$wm1" -v b="$v1" 'BEGIN{if (a-b>1e-5||b-a>1e-5) exit 1}' || { echo "FAIL: writer mean $wm1 != $v1"; exit 2; }
  awk -v a="$rm1" -v b="$v1" 'BEGIN{if (a-b>1e-5||b-a>1e-5) exit 1}' || { echo "FAIL: reader mean $rm1 != $v1"; exit 3; }
fi

# Phase 2: reader writes v2, writer reads
v2=2.3456
echo "WRITE $v2" >&5
r_wrote_line=$(read_until_prefix 6 "WROTE" 10)

echo "READMEAN" >&3
w_mean2_line=$(read_until_prefix 4 "MEAN" 10)

rm2=${r_wrote_line#WROTE }
wm2=${w_mean2_line#MEAN }

if (( both_cuda == 1 )); then
  awk -v a="$rm2" -v b="$v2" 'BEGIN{if (a-b>1e-5||b-a>1e-5) exit 1}' || { echo "FAIL: reader mean $rm2 != $v2"; exit 4; }
  awk -v a="$wm2" -v b="$v2" 'BEGIN{if (a-b>1e-5||b-a>1e-5) exit 1}' || { echo "FAIL: writer mean $wm2 != $v2"; exit 5; }
fi

# Graceful shutdown
echo "QUIT" >&5
echo "QUIT" >&3

wait "$r_pid"
wait "$w_pid"

if (( both_cuda == 1 )); then
  echo "E2E PASS: posix_ipc mutex + CUDA IPC cross-process writes verified."
else
  echo "E2E SKIP (no CUDA): posix_ipc mutex exercised; CUDA sharing not enforced."
fi

