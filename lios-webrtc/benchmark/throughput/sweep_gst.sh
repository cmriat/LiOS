#!/usr/bin/env bash
# Rigorous gst throughput: sweep target FPS to saturation, N repeats per point,
# log GPU (overall/encoder/decoder) utilization, report mean±std.
# Localhost loopback. Finds the real max figures/sec (where achieved < target).
#
# Env: FPS_LIST REPEATS DURATION WARMUP W H CUDA PORT
set -u
ROOT="$(cd "$(dirname "$0")/../../" && pwd)"; cd "$ROOT"

FPS_LIST=${FPS_LIST:-"120 240 360 480 600"}
REPEATS=${REPEATS:-3}
DURATION=${DURATION:-8}; WARMUP=${WARMUP:-3}
W=${W:-640}; H=${H:-480}; CUDA=${CUDA:-1}
PORT=${PORT:-18093}
SIGSRV=${SIGSRV:-/tmp/sigsrv}
LOG=$(mktemp -d /tmp/sweep-gst.XXXX)

export NO_PROXY=127.0.0.1,localhost
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

cleanup(){
  pkill -f "$SIGSRV serve --addr :$PORT" 2>/dev/null
  pkill -f "throughput/gst_sender.py"    2>/dev/null
  pkill -f "throughput/gst_receiver.py"  2>/dev/null
  pkill -f "nvidia-smi --query-gpu"      2>/dev/null
}
trap cleanup EXIT

[ -x "$SIGSRV" ] || pixi run bash -c "cd signal-server && go build -o $SIGSRV ." || { echo build_fail; exit 1; }
"$SIGSRV" serve --addr ":$PORT" >"$LOG/sig.log" 2>&1 &
sleep 1

echo "=== gst 吞吐饱和扫描 · ${W}x${H} · CUDA=$CUDA · 每点 ${REPEATS} 次 · 测窗 ${DURATION}s ==="
printf "%-8s %-6s %-12s %-6s %-6s %-6s\n" target run figures/sec GPU% ENC% DEC%
SUMMARY="$LOG/summary.txt"; : > "$SUMMARY"

for fps in $FPS_LIST; do
  vals=""
  for r in $(seq 1 "$REPEATS"); do
    room="sw_${fps}_${r}"
    ROOM=$room SIGNAL_URL="ws://127.0.0.1:$PORT/ws" W=$W H=$H FPS=$fps \
      pixi run python benchmark/throughput/gst_sender.py >"$LOG/snd.log" 2>&1 &
    SND=$!
    sleep 2
    nvidia-smi --query-gpu=utilization.gpu,utilization.encoder,utilization.decoder \
      --format=csv,noheader,nounits -l 1 >"$LOG/gpu.csv" 2>/dev/null &
    NV=$!
    out=$(ROOM=$room SIGNAL_URL="ws://127.0.0.1:$PORT/ws" DURATION=$DURATION WARMUP=$WARMUP CUDA=$CUDA \
      timeout $((WARMUP+DURATION+25)) pixi run python benchmark/throughput/gst_receiver.py 2>/dev/null \
      | grep -a RESULT_JSON | tail -1)
    kill $NV 2>/dev/null; kill $SND 2>/dev/null
    pkill -f "throughput/gst_sender.py" 2>/dev/null; sleep 1
    fpsval=$(echo "$out" | sed 's/^RESULT_JSON //' | python3 -c "import sys,json;print(json.load(sys.stdin)['fps'])" 2>/dev/null)
    [ -z "$fpsval" ] && fpsval=0
    read -r g e d <<<"$(awk -F, 'NF>=3{g+=$1;e+=$2;d+=$3;n++} END{if(n)printf "%.0f %.0f %.0f",g/n,e/n,d/n; else print "- - -"}' "$LOG/gpu.csv" 2>/dev/null)"
    printf "%-8s %-6s %-12s %-6s %-6s %-6s\n" "$fps" "r$r" "$fpsval" "$g" "$e" "$d"
    vals="$vals $fpsval"
  done
  python3 -c "
import statistics as st
v=[float(x) for x in '$vals'.split()]
print(f'  >>> target={$fps}: achieved mean={st.mean(v):.1f} std={st.pstdev(v):.2f} fps (n={len(v)})')
" | tee -a "$SUMMARY"
done

echo; echo "=== 汇总(找拐点:achieved 明显低于 target 即饱和)==="
cat "$SUMMARY"
echo "logs: $LOG"
