#!/usr/bin/env bash
# LiveKit Cloud 发布端 (在 admin01 跑): 读公共片段 /tmp/clip_i420.raw 推到 LiveKit Cloud,
# 供 remote 上的 livekit_subscriber.py 跨机测吞吐。先起 remote 订阅端, 再跑这个。
#
# 用法(可用 env 覆盖):
#   benchmark/livekit/cloud_pub.sh
#   FPS=600 MAXBR=2000000 ROOM=xmcloud DURATION=40 benchmark/livekit/cloud_pub.sh
set -u
cd "$(cd "$(dirname "$0")/../.." && pwd)"   # 切到仓库根

# Cloud 凭据(.env 里的 LK_URL/LK_KEY/LK_SECRET)
[ -f .env ] || { echo "缺 .env (Cloud 凭据)"; exit 1; }
set -a; source .env; set +a

# 关键: 大小写代理都要清, 否则 WS 握手失败 (HandshakeIncomplete)
export NO_PROXY=127.0.0.1,localhost,.livekit.cloud,mylive-lfsyb4uy.livekit.cloud
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

FPS=${FPS:-400}              # 目标帧率(同时设为 max_framerate)
MAXBR=${MAXBR:-10000000}     # 码率上限(bps); 和 gst 对齐就改这个 (如 2000000=2Mbps)
ROOM=${ROOM:-xmcloud}        # 必须和 remote 订阅端一致
DURATION=${DURATION:-40}     # 推流时长(秒), 要够订阅端 WARMUP+DURATION
W=${W:-640}; H=${H:-480}
CLIP=${CLIP:-/tmp/clip_i420.raw}

[ -f "$CLIP" ] || { echo "缺公共片段 $CLIP — 先跑: pixi run python benchmark/throughput/gen_clip.py"; exit 1; }

echo "[cloud-pub] URL=$LK_URL room=$ROOM target=${FPS}fps MAXBR=$((MAXBR/1000000))Mbps clip=$CLIP dur=${DURATION}s"
LK_URL="$LK_URL" LK_KEY="$LK_KEY" LK_SECRET="$LK_SECRET" LK_ROOM="$ROOM" \
  W="$W" H="$H" FPS="$FPS" MAXFPS="$FPS" MAXBR="$MAXBR" CLIP="$CLIP" DURATION="$DURATION" \
  pixi run -e livekit python benchmark/livekit/livekit_publisher.py
