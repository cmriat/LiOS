#!/usr/bin/env bash
set -euo pipefail

dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# Working dir for logs
workdir=$(mktemp -d -t ipc2pflask-XXXXXX)
cleanup() {
  set +e
  [[ -d "$workdir" ]] && rm -rf "$workdir" || true
}
trap cleanup EXIT

writer_log="$workdir/writer.log"
reader_log="$workdir/reader.log"
mkdir -p "$workdir"
: >"$writer_log"
: >"$reader_log"

read_line_timeout() {
  local fd="$1"; shift
  local timeout_sec="${1:-20}"; shift || true
  local line
  if IFS= read -r -t "$timeout_sec" -u "$fd" line; then
    printf '%s' "$line"
    return 0
  fi
  return 1
}

read_until_prefix() {
  local fd="$1"; local prefix="$2"; local timeout_sec="${3:-30}"
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

# Find a free TCP port
pick_port() {
  python - "$@" <<'PY'
import socket, sys
s=socket.socket()
s.bind(("127.0.0.1",0))
port=s.getsockname()[1]
print(port)
s.close()
PY
}

PORT=$(pick_port)
HOST=127.0.0.1
BASE_URL="http://$HOST:$PORT"

echo "LOGDIR $workdir"

# Start writer
(
  set -eo pipefail
  pixi run python "$dir/writer.py" --host "$HOST" --port "$PORT"
) >"$writer_log" 2>&1 &
w_pid=$!

# Tail writer logs on FD 4 (open immediately; file will grow)
exec 4<"$writer_log"

# Wait for writer readiness
ready_line=$(read_until_prefix 4 "READY" 30)
server_line=$(read_until_prefix 4 "SERVER" 30)

echo "writer: $ready_line"
echo "writer: $server_line"

# Start reader
(
  set -eo pipefail
  pixi run python "$dir/reader.py" --url "$BASE_URL"
) >"$reader_log" 2>&1 &
r_pid=$!

# Tail reader logs on FD 6
exec 6<"$reader_log"

r_ready_line=$(read_until_prefix 6 "READY" 30)
echo "reader: $r_ready_line"

# Determine CUDA availability on both ends
both_cuda=0
if [[ "$ready_line" == *"cuda=1"* && "$r_ready_line" == *"cuda=1"* ]]; then
  both_cuda=1
fi

# Expect reader writes V2 and writer observes it
obs_line=$(read_until_prefix 4 "OBSERVED_V2" 30 || true)
if (( both_cuda == 1 )); then
  if [[ -z "$obs_line" ]]; then
    echo "FAIL: writer did not observe reader's v2 within timeout" >&2
    echo "Writer log: $writer_log" >&2
    echo "Reader log: $reader_log" >&2
    kill "$r_pid" "$w_pid" 2>/dev/null || true
    wait "$r_pid" 2>/dev/null || true
    wait "$w_pid" 2>/dev/null || true
    exit 4
  fi
fi

# Wait processes
wait "$r_pid"
stop_line=$(read_until_prefix 4 "SERVER_STOPPED" 10 || true)
wait "$w_pid"

# Post-stop health check (best-effort): should be unreachable
python - "$BASE_URL" <<'PY' || true
import sys, urllib.request, urllib.error
base=sys.argv[1].rstrip('/')
u=f"{base}/api/v1/healthz"
try:
    urllib.request.urlopen(u, timeout=1.5)
    print("WARN: healthz still reachable after server stop (non-fatal)")
except Exception:
    print("HEALTHZ_UNREACHABLE_AFTER_STOP")
PY

ts_dir="$dir/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$ts_dir"
cp -f "$writer_log" "$ts_dir/writer.log" || true
cp -f "$reader_log" "$ts_dir/reader.log" || true

if (( both_cuda == 1 )); then
  echo "E2E PASS: Flask ok + posix_ipc mutex + CUDA IPC works; logs under $ts_dir"
  echo "PASS" >"$ts_dir/summary.txt"
else
  echo "E2E SKIP (no CUDA): Flask ok + posix_ipc exercised; CUDA checks skipped; logs under $ts_dir"
  echo "SKIP_NO_CUDA" >"$ts_dir/summary.txt"
fi
