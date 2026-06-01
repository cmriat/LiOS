#!/usr/bin/env bash
# LiveKit native (Rust libwebrtc) throughput via public SFU. pub+sub on this machine
# (both libwebrtc, no Python GIL) -> isolates the codec ceiling vs Python SDK's 25fps.
set -u
cd /home/admin01/gst-webrtc
[ -f dev/secrets.env ] && . dev/secrets.env   # real creds live in dev/ (gitignored; scp between machines)
BIN=benchmark/livekit/lk_rust/target/release/lk_rust
URL=ws://RELAY_HOST:7880; KEY=${KEY:-devkey}
SECRET=${SECRET:?missing LK secret — put it in dev/secrets.env (scp from the sender machine)}
ROOM=${ROOM:-rusttp}
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null||true
export NO_PROXY=RELAY_HOST,127.0.0.1,localhost
[ -x "$BIN" ] || { echo "binary 未就绪: $BIN"; exit 1; }

TPUB=$(pixi run -e livekit python /tmp/mktoken.py "$KEY" "$SECRET" "$ROOM" rustpub 2>/dev/null | tail -1)
TSUB=$(pixi run -e livekit python /tmp/mktoken.py "$KEY" "$SECRET" "$ROOM" rustsub 2>/dev/null | tail -1)
echo "tokens: pub=${#TPUB}b sub=${#TSUB}b"

# subscriber first
LK_URL=$URL LK_TOKEN=$TSUB MODE=sub WARMUP=4 DURATION=12 "$BIN" >/tmp/rust_sub.log 2>&1 &
SUB=$!; sleep 4
# publisher push hard
LK_URL=$URL LK_TOKEN=$TPUB MODE=pub W=640 H=480 FPS=300 DURATION=20 "$BIN" >/tmp/rust_pub.log 2>&1 &
PUB=$!
wait "$SUB"; kill "$PUB" 2>/dev/null
echo "--- sub 结果 ---"; grep -a RESULT_JSON /tmp/rust_sub.log | tail -1
echo "--- pub 末尾 ---"; tail -2 /tmp/rust_pub.log
echo "--- sub 末尾(诊断) ---"; tail -3 /tmp/rust_sub.log
echo "=== done ==="
