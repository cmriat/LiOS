#!/usr/bin/env bash
# LiveKit-native bidirectional latency (NTP 4-ts over media) -> true one-way, offset cancelled.
# FWD admin01 pub -> remote sub ; REV remote pub -> admin01 sub. Both via public SFU.
set -u
cd /home/admin01/gst-webrtc
[ -f dev/secrets.env ] && . dev/secrets.env   # real creds live in dev/ (gitignored; scp between machines)
URL=ws://RELAY_HOST:7880; KEY=${KEY:-devkey}
SECRET=${SECRET:?missing LK secret — put it in dev/secrets.env (scp from the sender machine)}
REMOTE=your-remote-host; RDIR=~/gst-webrtc
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null||true
export NO_PROXY=RELAY_HOST,your-host.example.com,.example.com,127.0.0.1,localhost
ENVV="LK_URL=$URL LK_KEY=$KEY LK_SECRET='$SECRET' LK_FORCE_RELAY=0"
SND=benchmark/livekit/livekit_sender_ts.py; RCV=benchmark/livekit/livekit_receiver_ts.py
cleanup(){ pkill -f livekit_sender_ts.py 2>/dev/null; pkill -f livekit_receiver_ts.py 2>/dev/null; }
trap cleanup EXIT

pair(){ python3 - "$1" "$2" <<'PY'
import sys,bisect,statistics as st
send=sorted(float(x) for x in open(sys.argv[1]) if x.strip())
recv=sorted(float(x) for x in open(sys.argv[2]) if x.strip())
if not send or not recv: print("NA 0"); sys.exit()
lat=[(tr-send[bisect.bisect_right(send,tr)-1])*1000 for tr in recv if bisect.bisect_right(send,tr)>0]
s=lat[1:]; print(f"{st.median(s):.2f} {len(s)}" if s else "NA 0")
PY
}

echo "=== FWD admin01 -> remote (LiveKit) ==="
R=lkb_f
ssh "$REMOTE" "cd $RDIR; unset http_proxy https_proxy all_proxy 2>/dev/null;
  $ENVV LK_ROOM=$R DURATION=26 pixi run -e livekit python $RCV >/tmp/lkb_f_recv.log 2>&1" &
S=$!; sleep 7
LK_URL=$URL LK_KEY=$KEY LK_SECRET="$SECRET" LK_FORCE_RELAY=0 LK_ROOM=$R FPS=1 DURATION=22 \
  pixi run -e livekit python $SND >/tmp/lkb_f_send.log 2>&1 &
wait $!; wait "$S" 2>/dev/null; cleanup
grep -aoE '\[ts\]\[lk-sender\]\[cam0\] [0-9.]+' /tmp/lkb_f_send.log|awk '{print $NF}'>/tmp/lkb_f_send.txt
ssh "$REMOTE" 'grep -aoE "\[ts\]\[lk-receiver\]\[cam0\] [0-9.]+" /tmp/lkb_f_recv.log'|awk '{print $NF}'>/tmp/lkb_f_recv.txt
read MF NF < <(pair /tmp/lkb_f_send.txt /tmp/lkb_f_recv.txt); echo "  M_f=${MF}ms (n=$NF)"
sleep 3

echo "=== REV remote -> admin01 (LiveKit) ==="
R=lkb_r
PYTHONUNBUFFERED=1 LK_URL=$URL LK_KEY=$KEY LK_SECRET="$SECRET" LK_FORCE_RELAY=0 LK_ROOM=$R DURATION=30 \
  pixi run -e livekit python $RCV >/tmp/lkb_r_recv.log 2>&1 &
RPID=$!; sleep 7
ssh "$REMOTE" "cd $RDIR; unset http_proxy https_proxy all_proxy 2>/dev/null;
  $ENVV LK_ROOM=$R FPS=1 DURATION=22 pixi run -e livekit python $SND >/tmp/lkb_r_send.log 2>&1" &
wait $!; sleep 2; kill "$RPID" 2>/dev/null; sleep 1; cleanup
grep -aoE '\[ts\]\[lk-receiver\]\[cam0\] [0-9.]+' /tmp/lkb_r_recv.log|awk '{print $NF}'>/tmp/lkb_r_recv.txt
ssh "$REMOTE" 'grep -aoE "\[ts\]\[lk-sender\]\[cam0\] [0-9.]+" /tmp/lkb_r_send.log'|awk '{print $NF}'>/tmp/lkb_r_send.txt
read MB NB < <(pair /tmp/lkb_r_send.txt /tmp/lkb_r_recv.txt); echo "  M_b=${MB}ms (n=$NB)"

python3 - "$MF" "$MB" <<'PY'
import sys
mf,mb=sys.argv[1],sys.argv[2]
if "NA" in (mf,mb): print(f"无法合成 (M_f={mf} M_b={mb})"); sys.exit()
mf=float(mf);mb=float(mb)
print(f"\n>>> LiveKit 真实单向 d=(M_f+M_b)/2={(mf+mb)/2:.2f}ms  时钟偏移={(mf-mb)/2:.2f}ms")
print(f"RESULT_D={(mf+mb)/2:.2f}")
PY
echo "=== done ==="
