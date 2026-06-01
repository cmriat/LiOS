#!/usr/bin/env bash
# Cross-machine gst P2P throughput sweep: admin01 sender -> public coturn -> remote
# receiver (C-side fps). Finds the real cross-network saturation point.
set -u
cd /home/admin01/gst-webrtc
SIG=${SIG:-ws://RELAY_HOST:18080/ws}
STUN=${STUN:-stun://RELAY_HOST:3478}
TURN=${TURN:-turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp}
REMOTE=${REMOTE:-your-remote-host}
FPS_LIST=${FPS_LIST:-"300 600 900 1200 1800"}
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=RELAY_HOST,your-host.example.com,.example.com,127.0.0.1,localhost
trap 'pkill -f throughput/gst_sender.py 2>/dev/null' EXIT

echo "=== gst P2P 跨机扫描 (admin01 -> coturn RELAY_HOST -> remote-gpu) ==="
for fps in $FPS_LIST; do
  room=p2psw$fps
  ROOM=$room SIGNAL_URL=$SIG STUN=$STUN TURN="$TURN" W=640 H=480 FPS=$fps \
    pixi run python benchmark/throughput/gst_sender.py >/tmp/p2psnd.log 2>&1 &
  sleep 3
  out=$(ssh "$REMOTE" "cd ~/gst-webrtc && unset http_proxy https_proxy all_proxy 2>/dev/null; ROOM=$room SIGNAL_URL=$SIG STUN=$STUN TURN='$TURN' DURATION=12 WARMUP=3 pixi run python benchmark/throughput/gst_receiver.py 2>/dev/null | grep -a RESULT_JSON | tail -1")
  fpsval=$(echo "$out" | sed 's/^RESULT_JSON //' | python3 -c "import sys,json;print(json.load(sys.stdin).get('fps',0))" 2>/dev/null)
  echo "  P2P target=$fps -> figures/sec=${fpsval:-0}"
  pkill -f throughput/gst_sender.py 2>/dev/null; sleep 2
done
echo "=== done ==="
