#!/usr/bin/env bash
# Parametrized LiveKit bidirectional latency (NTP 4-timestamp -> true one-way,
# clock offset cancelled). Reuses livekit_sender_ts.py / livekit_receiver_ts.py
# (glass -> RGBA -> CUDA tensor -> cuda.synchronize() measurement point, 1 fps).
#
#   FWD admin01(pub) -> SFU -> pod(sub):  M_f = recv_pod - send_admin = d + offset
#   REV pod(pub)     -> SFU -> admin01(sub): M_b = recv_admin - send_pod = d - offset
#   true one-way d = (M_f + M_b)/2 ; clock offset = (M_f - M_b)/2
#
# Env: LK_URL LK_KEY LK_SECRET  [LABEL] [LK_FORCE_RELAY]
set -u
cd /home/admin01/gst-webrtc
: "${LK_URL:?need LK_URL}"; : "${LK_KEY:?need LK_KEY}"; : "${LK_SECRET:?need LK_SECRET}"
LABEL=${LABEL:-lk}
export LK_FORCE_RELAY=${LK_FORCE_RELAY:-0}
REMOTE=your-remote-host
RDIR=~/gst-webrtc
SND=benchmark/livekit/livekit_sender_ts.py
RCV=benchmark/livekit/livekit_receiver_ts.py
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=RELAY_HOST,your-host.example.com,.example.com,.livekit.cloud,127.0.0.1,localhost
export LK_URL LK_KEY LK_SECRET

REMENV="LK_URL='$LK_URL' LK_KEY='$LK_KEY' LK_SECRET='$LK_SECRET' LK_FORCE_RELAY=$LK_FORCE_RELAY"
cleanup(){
  pkill -f livekit_sender_ts.py 2>/dev/null; pkill -f livekit_receiver_ts.py 2>/dev/null
  ssh -o BatchMode=yes "$REMOTE" 'pkill -f livekit_sender_ts.py; pkill -f livekit_receiver_ts.py' 2>/dev/null
}
trap cleanup EXIT

pair(){ python3 - "$1" "$2" <<'PY'
import sys,bisect,statistics as st
send=sorted(float(x) for x in open(sys.argv[1]) if x.strip())
recv=sorted(float(x) for x in open(sys.argv[2]) if x.strip())
if not send or not recv: print("NA 0"); sys.exit()
lat=[(tr-send[bisect.bisect_right(send,tr)-1])*1000 for tr in recv if bisect.bisect_right(send,tr)>0]
s=lat[1:]
print(f"{st.median(s):.2f} {len(s)}" if s else "NA 0")
PY
}

echo "=== [$LABEL] FWD admin01 -> pod  (URL=$LK_URL) ==="
R=${LABEL}_f
ssh -o BatchMode=yes "$REMOTE" "cd $RDIR; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null;
  $REMENV LK_ROOM=$R DURATION=26 pixi run -e livekit python $RCV >/tmp/lat_f_recv.log 2>&1" &
S=$!; sleep 9
LK_ROOM=$R FPS=1 DURATION=22 pixi run -e livekit python $SND >/tmp/lat_f_send.log 2>&1 &
wait $!; wait "$S" 2>/dev/null; cleanup
grep -aoE '\[ts\]\[lk-sender\]\[cam0\] [0-9.]+' /tmp/lat_f_send.log | awk '{print $NF}' > /tmp/lat_f_send.txt
ssh -o BatchMode=yes "$REMOTE" 'grep -aoE "\[ts\]\[lk-receiver\]\[cam0\] [0-9.]+" /tmp/lat_f_recv.log' | awk '{print $NF}' > /tmp/lat_f_recv.txt
read MF NF < <(pair /tmp/lat_f_send.txt /tmp/lat_f_recv.txt)
echo "  M_f=${MF}ms (n=$NF)  [send=$(wc -l </tmp/lat_f_send.txt) recv=$(wc -l </tmp/lat_f_recv.txt)]"
sleep 3

echo "=== [$LABEL] REV pod -> admin01 ==="
R=${LABEL}_r
PYTHONUNBUFFERED=1 LK_ROOM=$R DURATION=30 pixi run -e livekit python $RCV >/tmp/lat_r_recv.log 2>&1 &
RPID=$!; sleep 9
ssh -o BatchMode=yes "$REMOTE" "cd $RDIR; unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null;
  $REMENV LK_ROOM=$R FPS=1 DURATION=22 pixi run -e livekit python $SND >/tmp/lat_r_send.log 2>&1" &
wait $!; sleep 2; kill "$RPID" 2>/dev/null; sleep 1; cleanup
grep -aoE '\[ts\]\[lk-receiver\]\[cam0\] [0-9.]+' /tmp/lat_r_recv.log | awk '{print $NF}' > /tmp/lat_r_recv.txt
ssh -o BatchMode=yes "$REMOTE" 'grep -aoE "\[ts\]\[lk-sender\]\[cam0\] [0-9.]+" /tmp/lat_r_send.log' | awk '{print $NF}' > /tmp/lat_r_send.txt
read MB NB < <(pair /tmp/lat_r_send.txt /tmp/lat_r_recv.txt)
echo "  M_b=${MB}ms (n=$NB)  [send=$(wc -l </tmp/lat_r_send.txt) recv=$(wc -l </tmp/lat_r_recv.txt)]"

python3 - "$MF" "$MB" "$LABEL" <<'PY'
import sys
mf,mb,lab=sys.argv[1],sys.argv[2],sys.argv[3]
if "NA" in (mf,mb): print(f"\n>>> [{lab}] 无法合成单向延迟 (M_f={mf} M_b={mb}) — 看上面 send/recv 计数"); sys.exit()
mf=float(mf);mb=float(mb)
print(f"\n>>> [{lab}] LiveKit 真实单向延迟 d=(M_f+M_b)/2 = {(mf+mb)/2:.2f} ms   (时钟偏移 {(mf-mb)/2:.2f} ms)")
print(f"RESULT_D={(mf+mb)/2:.2f}")
PY
echo "=== [$LABEL] done ==="
