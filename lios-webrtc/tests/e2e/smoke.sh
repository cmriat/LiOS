#!/usr/bin/env bash
# Minimal end-to-end smoke test orchestrator
# - Starts Go signal server on :18080
# - Runs Python receiver.py and sender_sw.py (software x264enc) via pixi
# - Watches logs for success patterns
# - Writes e2e/report.json and exits non-zero on failure

set -Eeuo pipefail
IFS=$'\n\t'

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
E2E_DIR="$ROOT_DIR/tests/e2e"
LOG_DIR="$E2E_DIR/logs"
REPORT_JSON="$E2E_DIR/report.json"

mkdir -p "$LOG_DIR"

PORT=18080
SIGNAL_URL="ws://127.0.0.1:${PORT}/ws"
ROOM="demo"
TIMEOUT_SEC=${TIMEOUT_SEC:-25}

server_pid=""; sender_pid=""; receiver_pid=""

cleanup() {
  # Best-effort shutdown in reverse order
  for pid in "$sender_pid" "$receiver_pid" "$server_pid"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
      # escalate if stubborn
      sleep 0.5 || true
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

ts_ns() { date +%s%N; }
start_ns=$(ts_ns)

wait_port() {
  local host=$1 port=$2 deadline=$((SECONDS+TIMEOUT_SEC))
  pixi run python - "$host" "$port" <<'PY'
import socket, sys, time
host, port = sys.argv[1], int(sys.argv[2])
deadline = time.time() + 30.0  # safety; shell controls outer timeout
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            sys.exit(0)
    except OSError:
        time.sleep(0.05)
sys.exit(1)
PY
}

echo "[e2e] starting signal-server on :${PORT}"
(
  cd "$ROOT_DIR/signal-server"
  # Start as background job of parent so $! is set correctly
  GO111MODULE=on exec go run . serve --addr ":${PORT}" \
    >"$LOG_DIR/signal.log" 2>&1
) &
server_pid=$!

if ! wait_port 127.0.0.1 "$PORT"; then
  echo "[e2e][FATAL] signal-server did not start listening on :$PORT" | tee -a "$LOG_DIR/signal.log"
  exit 2
fi
echo "[e2e] signal-server is up (pid=$server_pid)"

echo "[e2e] starting receiver.py (headless)"
(
  cd "$ROOT_DIR"
  ROOM="$ROOM" SIGNAL_URL="$SIGNAL_URL" PYTHONUNBUFFERED=1 \
    exec pixi run python -u tests/e2e/receiver.py \
    >"$LOG_DIR/receiver.log" 2>&1
) &
receiver_pid=$!

# Give receiver a moment to join room
sleep 1

echo "[e2e] starting sender_sw.py"
(
  cd "$ROOT_DIR"
  # Software x264enc sender: no GPU or camera dependency, runs anywhere
  ROOM="$ROOM" SIGNAL_URL="$SIGNAL_URL" PYTHONUNBUFFERED=1 \
    exec pixi run python -u tests/e2e/sender_sw.py \
    >"$LOG_DIR/sender.log" 2>&1
) &
sender_pid=$!

# ---- Wait for success patterns ----
echo "[e2e] waiting for link/connect signals (timeout ${TIMEOUT_SEC}s)"

until [[ $SECONDS -ge $TIMEOUT_SEC ]]; do
  sleep 0.2
  if [[ -f "$LOG_DIR/receiver.log" ]] && grep -q "\[receiver\] linked webrtc src to sink bin" "$LOG_DIR/receiver.log"; then
    receiver_ok=1
  else
    receiver_ok=0
  fi
  if [[ -f "$LOG_DIR/sender.log" ]] && (grep -q "\[webrtc\] set remote description (answer)" "$LOG_DIR/sender.log" || grep -q "connection state: connected" "$LOG_DIR/sender.log"); then
    sender_ok=1
  else
    sender_ok=0
  fi
  if [[ $receiver_ok -eq 1 && $sender_ok -eq 1 ]]; then
    break
  fi
done

success=0
if [[ ${receiver_ok:-0} -eq 1 && ${sender_ok:-0} -eq 1 ]]; then
  success=1
  echo "[e2e][PASS] sender and receiver linked via signaling"
else
  echo "[e2e][FAIL] conditions not met within ${TIMEOUT_SEC}s"
  echo "---- sender.log (tail) ----"; tail -n 100 "$LOG_DIR/sender.log" || true
  echo "---- receiver.log (tail) ----"; tail -n 100 "$LOG_DIR/receiver.log" || true
  echo "---- signal.log (tail) ----"; tail -n 60 "$LOG_DIR/signal.log" || true
fi

end_ns=$(ts_ns)
dur_ms=$(( (end_ns - start_ns)/1000000 ))

appsink_ok=0
cam0_count=0
cam1_count=0

# If first stage passed, run appsink msid e2e as stage 2 (separate room to avoid interference)
if [[ ${success} -eq 1 ]]; then
  echo "[e2e] running appsink-msid stage"
  FRAMES_DIR="$E2E_DIR/frames_msid"
  rm -rf "$FRAMES_DIR" && mkdir -p "$FRAMES_DIR"
  MSID_ROOM="${ROOM}-msid-$$"
  if ROOM="$MSID_ROOM" SIGNAL_URL="$SIGNAL_URL" \
    pixi run python -u "$ROOT_DIR/tests/e2e/appsink_msid_e2e.py" \
      --names cam0 cam1 --frames 2 --out "$FRAMES_DIR" \
      >"$LOG_DIR/appsink_msid.log" 2>&1; then
    # Count frames per stream to validate naming + image output
    cam0_count=$(ls -1 "$FRAMES_DIR"/cam0_*.png 2>/dev/null | wc -l | sed 's/ //g')
    cam1_count=$(ls -1 "$FRAMES_DIR"/cam1_*.png 2>/dev/null | wc -l | sed 's/ //g')
    if [[ ${cam0_count} -ge 1 && ${cam1_count} -ge 1 ]]; then
      appsink_ok=1
      echo "[e2e][PASS] appsink-msid stage (cam0=${cam0_count}, cam1=${cam1_count})"
    else
      echo "[e2e][FAIL] appsink-msid images missing (cam0=${cam0_count}, cam1=${cam1_count})"
      echo "---- appsink_msid.log (tail) ----"; tail -n 120 "$LOG_DIR/appsink_msid.log" || true
    fi
  else
    echo "[e2e][FAIL] appsink-msid script failed"
    echo "---- appsink_msid.log (tail) ----"; tail -n 120 "$LOG_DIR/appsink_msid.log" || true
  fi
fi

# Overall success requires both stages
overall=$(( success == 1 && appsink_ok == 1 ))

# Write report (combined)
cat >"$REPORT_JSON" <<JSON
{
  "signal_url": "${SIGNAL_URL}",
  "room": "${ROOM}",
  "receiver_linked": ${receiver_ok:-0},
  "sender_connected": ${sender_ok:-0},
  "appsink_msid": ${appsink_ok},
  "appsink_cam0_frames": ${cam0_count},
  "appsink_cam1_frames": ${cam1_count},
  "success": ${overall},
  "duration_ms": ${dur_ms}
}
JSON

if [[ ${overall} -ne 1 ]]; then
  exit 1
fi

exit 0
