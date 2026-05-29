#!/usr/bin/env bash
# E2E test for gpu_sink_save.py: verifies frames are saved

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
E2E_DIR="$ROOT_DIR/e2e"
LOG_DIR="$E2E_DIR/logs"
ART_DIR="$E2E_DIR/artifacts"
OUT_DIR="$ART_DIR/gpu_frames"
REPORT_JSON="$E2E_DIR/gpu_sink_report.json"

mkdir -p "$LOG_DIR" "$ART_DIR" "$OUT_DIR"

PORT=${PORT:-18080}
SIGNAL_URL=${SIGNAL_URL:-"ws://127.0.0.1:${PORT}/ws"}
ROOM=${ROOM:-demo}
FRAMES=${FRAMES:-5}
TIMEOUT_SEC=${TIMEOUT_SEC:-40}

server_pid=""
sender_pid=""
receiver_pid=""

cleanup() {
  for pid in "$sender_pid" "$receiver_pid" "$server_pid"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      sleep 0.3 || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

ts_ns() { date +%s%N; }
start_ns=$(ts_ns)

wait_port() {
  local host=$1 port=$2 deadline=$((SECONDS + TIMEOUT_SEC))
  pixi run python - "$host" "$port" <<'PY'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
deadline = time.time() + 30.0
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            sys.exit(0)
    except OSError:
        time.sleep(0.05)
sys.exit(1)
PY
}

# Reuse existing server if port is already open; otherwise start a new one
if wait_port 127.0.0.1 "$PORT"; then
  echo "[e2e-gpu] reusing existing signal-server on :${PORT}"
else
  echo "[e2e-gpu] starting signal-server on :${PORT}"
  (
    cd "$ROOT_DIR/signal-server"
    GO111MODULE=on exec go run . serve --addr ":${PORT}" \
      >"$LOG_DIR/signal_gpu.log" 2>&1
  ) &
  server_pid=$!
  if ! wait_port 127.0.0.1 "$PORT"; then
    echo "[e2e-gpu][FATAL] signal-server not listening on :$PORT" | tee -a "$LOG_DIR/signal_gpu.log"
    exit 2
  fi
  echo "[e2e-gpu] signal-server up (pid=$server_pid)"
fi

echo "[e2e-gpu] starting gpu_sink_save.py (frames=$FRAMES)"
(
  cd "$ROOT_DIR"
  ROOM="$ROOM" SIGNAL_URL="$SIGNAL_URL" PYTHONUNBUFFERED=1 \
    exec pixi run python -u tests/e2e/gpu_sink_save.py --frames "$FRAMES" --out "$OUT_DIR" \
    >"$LOG_DIR/gpu_sink_receiver.log" 2>&1
) &
receiver_pid=$!

sleep 1

echo "[e2e-gpu] starting sender_sw.py (x264enc)"
(
  cd "$ROOT_DIR"
  ROOM="$ROOM" SIGNAL_URL="$SIGNAL_URL" PYTHONUNBUFFERED=1 \
    exec pixi run python -u tests/e2e/sender_sw.py \
    >"$LOG_DIR/gpu_sink_sender.log" 2>&1
) &
sender_pid=$!

echo "[e2e-gpu] waiting for $FRAMES frames (timeout ${TIMEOUT_SEC}s)"

deadline=$((SECONDS + TIMEOUT_SEC))
frames_ok=0
receiver_done=0

shopt -s nullglob
while [[ $SECONDS -lt $deadline ]]; do
  files=("$OUT_DIR"/*.png)
  count=${#files[@]}
  if [[ $count -ge $FRAMES ]]; then
    frames_ok=1
    break
  fi
  if [[ -n "$receiver_pid" ]] && ! kill -0 "$receiver_pid" 2>/dev/null; then
    receiver_done=1
    break
  fi
  sleep 0.2
done

success=0
msg=""

if [[ $frames_ok -eq 1 ]]; then
  # Verify image integrity with PIL (if available in env)
  echo "[e2e-gpu] verifying saved images via PIL"
  verify_py=$(
    cat <<'PY'
import sys, glob
from PIL import Image
files = sorted(glob.glob(sys.argv[1]))
ok = 0
for f in files:
    try:
        with Image.open(f) as im:
            im.verify()
        ok += 1
    except Exception as e:
        print("verify-fail:", f, e)
        pass
print(ok)
PY
  )
  okn=$(
    pixi run python - "$OUT_DIR/*.png" <<PY 2>/dev/null | tail -n 1 | tr -cd '0-9'
$verify_py
PY
  )
  okn=${okn:-0}
  if [[ $okn -ge $FRAMES ]]; then
    success=1
    msg="frames_saved_and_verified"
  else
    msg="frames_count_ok_but_verify_failed"
  fi
else
  echo "---- gpu_sink_receiver.log (tail) ----"
  tail -n 120 "$LOG_DIR/gpu_sink_receiver.log" || true
  echo "---- gpu_sink_sender.log (tail) ----"
  tail -n 120 "$LOG_DIR/gpu_sink_sender.log" || true
  echo "---- signal_gpu.log (tail) ----"
  tail -n 80 "$LOG_DIR/signal_gpu.log" || true
  if [[ $receiver_done -eq 1 ]]; then
    msg="receiver_exited_early"
  else
    msg="timeout_waiting_frames"
  fi
fi

end_ns=$(ts_ns)
dur_ms=$(((end_ns - start_ns) / 1000000))

cat >"$REPORT_JSON" <<JSON
{
  "signal_url": "${SIGNAL_URL}",
  "room": "${ROOM}",
  "frames_requested": ${FRAMES},
  "out_dir": "${OUT_DIR}",
  "success": ${success},
  "message": "${msg}",
  "duration_ms": ${dur_ms}
}
JSON

if [[ $success -ne 1 ]]; then
  echo "[e2e-gpu][FAIL] ${msg}"
  exit 1
fi

echo "[e2e-gpu][PASS] ${msg} (${FRAMES} frames)"
exit 0
