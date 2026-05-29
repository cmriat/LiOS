#!/usr/bin/env bash
# 在 remote 上跑: 从 LiveKit Cloud 订阅 admin01 推的同一片段, 逐帧硬数交付吞吐。
# 跨机: admin01(cloud_pub.sh) -> LiveKit Cloud -> remote(本脚本)。
# 用 pixi 默认 env 的 python(remote 旧 repo 无 livekit feature, 故 pip 装进默认 env)。
# 凭据从同目录 lk_cloud.env 读(scp 过来的, 含 Cloud key/secret)。
# 用法:  [ROOM=xmcloud] [WARMUP=5] [DURATION=10] [PIXI_DIR=~/gst-webrtc]  bash remote_sub.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PIXI_DIR=${PIXI_DIR:-$HOME/gst-webrtc}   # pixi 项目目录(有 pixi.toml)

# 关代理(remote 有 VSCODE_PROXY_URI), 否则 WS 握手失败
export NO_PROXY=.livekit.cloud,127.0.0.1,localhost
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY VSCODE_PROXY_URI 2>/dev/null || true

# Cloud 凭据
[ -f "$HERE/lk_cloud.env" ] && { set -a; source "$HERE/lk_cloud.env"; set +a; }
: "${LK_URL:?need LK_URL (放 lk_cloud.env 或 export)}"
: "${LK_KEY:?need LK_KEY}"
: "${LK_SECRET:?need LK_SECRET}"

ROOM=${ROOM:-xmcloud}; WARMUP=${WARMUP:-5}; DURATION=${DURATION:-10}

cd "$PIXI_DIR" || { echo "找不到 pixi 项目目录 $PIXI_DIR"; exit 1; }

# 确保 livekit + livekit-api 装在 pixi 默认 env(api 是单独的包, 给 token 用)
if ! pixi run python -c "from livekit import api, rtc" 2>/dev/null; then
  echo "[remote-sub] 在 pixi 默认 env 装 livekit + livekit-api ..."
  pixi run python -m pip install -q 'livekit>=1.0' livekit-api || { echo "装 livekit 失败"; exit 1; }
fi

echo "[remote-sub] <- $LK_URL room=$ROOM warmup=${WARMUP}s measure=${DURATION}s (等 admin01 发布端...)"
LK_URL="$LK_URL" LK_KEY="$LK_KEY" LK_SECRET="$LK_SECRET" LK_ROOM="$ROOM" WARMUP="$WARMUP" DURATION="$DURATION" \
  pixi run python "$HERE/livekit_subscriber.py"
