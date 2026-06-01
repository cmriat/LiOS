#!/usr/bin/env bash
# Single-clock-equivalent latency via NTP 4-timestamp method over the MEDIA path.
# Forward  (admin01->remote): M_f = recv_B - send_A = d + offset
# Reverse  (remote->admin01): M_b = recv_A - send_B = d - offset   (paths symmetric)
#   => true one-way d = (M_f+M_b)/2  (clock offset cancels), offset=(M_f-M_b)/2
set -u
cd /home/admin01/gst-webrtc
SIG=${SIG:-ws://RELAY_HOST:18080/ws}
STUN=${STUN:-stun://RELAY_HOST:3478}
TURN=${TURN:-turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp}
REMOTE=${REMOTE:-your-remote-host}
RDIR=~/gst-webrtc
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export NO_PROXY=RELAY_HOST,your-host.example.com,.example.com,127.0.0.1,localhost
SND=benchmark/two_camera_sender_1fps_ts.py
RCV=benchmark/two_camera_receiver_inferbuf_1fps_ts.py
cleanup(){ pkill -f two_camera_sender_1fps_ts.py 2>/dev/null; pkill -f two_camera_receiver_inferbuf_1fps_ts.py 2>/dev/null; }
trap cleanup EXIT

pair(){ # $1=send_ts $2=recv_ts  -> prints "mean n"
python3 - "$1" "$2" <<'PY'
import sys,bisect,statistics as st
send=sorted(float(x) for x in open(sys.argv[1]) if x.strip())
recv=sorted(float(x) for x in open(sys.argv[2]) if x.strip())
if not send or not recv: print("NA 0"); sys.exit()
lat=[(tr-send[bisect.bisect_right(send,tr)-1])*1000 for tr in recv if bisect.bisect_right(send,tr)>0]
s=lat[1:]  # drop first (connect/keyframe)
print(f"{st.median(s):.2f} {len(s)}" if s else "NA 0")
PY
}

# ---------- FORWARD: admin01 sender -> remote receiver ----------
echo "=== FWD admin01 -> remote ==="
ROOM=blat_f
ssh "$REMOTE" "cd $RDIR; unset http_proxy https_proxy all_proxy 2>/dev/null;
  ROOM=$ROOM SIGNAL_URL=$SIG STUN=$STUN TURN='$TURN' DURATION=30 \
  pixi run python $RCV --streams 1 >/tmp/blat_f_recv.log 2>&1" &
S=$!; sleep 6
ROOM=$ROOM SIGNAL_URL=$SIG STUN=$STUN TURN="$TURN" NUM_CAMS=1 DURATION=34 \
  pixi run python $SND >/tmp/blat_f_send.log 2>&1 &
P=$!; wait "$S"; wait "$P" 2>/dev/null; cleanup
grep -aoE '\[ts\]\[sender\]\[cam0\] [0-9.]+' /tmp/blat_f_send.log | awk '{print $NF}' > /tmp/blat_f_send.txt
ssh "$REMOTE" 'grep -aoE "\[ts\]\[receiver\]\[cam0\] [0-9.]+" /tmp/blat_f_recv.log' | awk '{print $NF}' > /tmp/blat_f_recv.txt
read MF NF < <(pair /tmp/blat_f_send.txt /tmp/blat_f_recv.txt)
echo "  M_f=${MF}ms (n=$NF)"
sleep 3

# ---------- REVERSE: remote sender -> admin01 receiver ----------
echo "=== REV remote -> admin01 ==="
ROOM=blat_r
PYTHONUNBUFFERED=1 ROOM=$ROOM SIGNAL_URL=$SIG STUN=$STUN TURN="$TURN" DURATION=34 \
  pixi run python $RCV --streams 1 >/tmp/blat_r_recv.log 2>&1 &
RPID=$!; sleep 6
ssh "$REMOTE" "cd $RDIR; unset http_proxy https_proxy all_proxy 2>/dev/null;
  ROOM=$ROOM SIGNAL_URL=$SIG STUN=$STUN TURN='$TURN' NUM_CAMS=1 DURATION=34 \
  pixi run python $SND >/tmp/blat_r_send.log 2>&1" &
S2=$!; wait "$S2"; sleep 1; kill "$RPID" 2>/dev/null; cleanup
grep -aoE '\[ts\]\[receiver\]\[cam0\] [0-9.]+' /tmp/blat_r_recv.log | awk '{print $NF}' > /tmp/blat_r_recv.txt
ssh "$REMOTE" 'grep -aoE "\[ts\]\[sender\]\[cam0\] [0-9.]+" /tmp/blat_r_send.log' | awk '{print $NF}' > /tmp/blat_r_send.txt
read MB NB < <(pair /tmp/blat_r_send.txt /tmp/blat_r_recv.txt)
echo "  M_b=${MB}ms (n=$NB)"

# ---------- combine ----------
python3 - "$MF" "$MB" <<'PY'
import sys
mf,mb=sys.argv[1],sys.argv[2]
if "NA" in (mf,mb): print(f"无法合成 (M_f={mf} M_b={mb})"); sys.exit()
mf=float(mf); mb=float(mb)
d=(mf+mb)/2; off=(mf-mb)/2
print(f"\n>>> gst 真实单向延迟 d = (M_f+M_b)/2 = {d:.2f} ms   [时钟偏移消除]")
print(f">>> 估计时钟偏移 (remote-admin01) = (M_f-M_b)/2 = {off:.2f} ms")
print(f"RESULT_D={d:.2f}")
PY
echo "=== done ==="
