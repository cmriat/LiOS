# LiOS 使用指南

把机器人现场的摄像头画面,用 GPU(或 CPU)编码 → 经 WebRTC 传输 → 在云端解码,落进
推理缓冲喂给模型,并可在浏览器里实时看画面。这份指南讲清楚:**端到端怎么跑、用真实相机、
HTTP 实时看画面、以及所有配置项**。

> 约定:本仓库环境由 [pixi](https://pixi.sh) 管理,所有 Python 命令都用 `pixi run` 前缀。

---

## 1. 它怎么工作(30 秒理解)

一共三个角色 + 一个外部中继:

```
   [边缘机:sender]                                  [云端机:receiver]
   摄像头 → 编码(NVENC/x264) ──┐               ┌── 解码(NVDEC/avdec)→ 推理缓冲 → 模型
                               │               │                         └→ HTTP 实时预览
                               └─► signal-server(信令握手)◄─┘
              \──── 媒体流经 TURN/coturn 中继(SRTP) ────/
```

- **signal-server**(Go):只做"握手中介",转发 SDP/ICE。两端必须都能连到它。
- **sender / receiver**(Python):真正收发视频的两端。
- **TURN/coturn**:转发媒体流的中继(你自己部署的近网中继是 LiOS 的核心特性)。

---

## 2. 准备环境

```bash
pixi install          # 安装所有依赖(GStreamer 1.26 / CUDA / PyTorch / Go ...)
```

需要的外部条件:
- 一台能跑 **signal-server** 的机器(两端都能访问它的 IP:端口)。
- 一台 **TURN/coturn** 服务器(跨机/跨网时几乎必须;同机 localhost 测试可不用)。
- **GPU 可选**:有 NVIDIA GPU + GStreamer nvcodec 插件(`nvh264enc`/`nvh264dec`/`cudaupload`)
  就走硬件编解码;没有也能用 CPU(`x264enc`/`avdec_h264`)跑通,见 [第 7 节](#7-没有-gpu纯-cpu-跑)。

---

## 3. 配置:`.env` 全部参数

把模板复制成 `.env`(已被 git 忽略,可放真实密钥),两个 example 启动时会自动加载它
(`gst_webrtc.load_env()`;**已存在的环境变量优先于 `.env`**):

```bash
cp .env.example .env
```

| 变量 | 作用 | 默认 |
|---|---|---|
| `ROOM` | 房间名,**两端必须一致**,靠它互相发现 | `demo` |
| `SIGNAL_URL` | 信令服务器 WebSocket 地址 | `ws://127.0.0.1:18080/ws` |
| `STUN` | STUN 服务器 | `stun://stun.l.google.com:19302` |
| `TURN` | TURN/coturn 中继 `turn://user:pass@host:port?transport=udp` | 占位符 |
| `VIDEO_SOURCE` | `test`(合成测试图,无需相机)\| `v4l2`(真实相机) | `test` |
| `CAMERAS` | 要开哪几路(见下) | `cam0,cam1` |
| `WIDTH` / `HEIGHT` | 采集分辨率 | `640` / `480` |
| `FPS` | 默认帧率(`CAMERAS` 里每路的 `@fps` 会覆盖它) | `30` |
| `ENCODER` | 发送端编码器:`auto` \| `nv`(NVENC)\| `sw`(x264,无需 GPU) | `auto` |
| `DECODER` | 接收端解码器:`auto` \| `nv`(NVDEC)\| `sw`(avdec_h264) | `auto` |
| `FLASK_HOST` / `FLASK_PORT` | 接收端 HTTP 服务监听地址/端口(预览+缓冲接口) | `127.0.0.1` / `5082` |
| `LK_URL/LK_KEY/LK_SECRET` | **可选**,仅 LiveKit 对比 benchmark 用,跑 LiOS 不需要 | 空 |

`CAMERAS` 格式:
- `test` 模式:逗号分隔的流名,可带帧率 → `cam0,cam1` 或 `cam0@30,cam1@15`
- `v4l2` 模式:`名字=/dev/videoN@帧率` → `mid=/dev/video0@30,left=/dev/video4@25`

`auto` 的含义:`ENCODER=auto` 在检测到 `nvh264enc`+`cudaupload` 时用 NVENC,否则自动退到
x264 软编;`DECODER=auto` 同理(有 `nvh264dec` 用 NVDEC,否则 `avdec_h264`)。

---

## 4. 跑通端到端

### 4.1 单机快速验证(最省事,先确认链路通)

同一台机器开三个终端(都在仓库根目录)。`.env` 用默认值即可(`SIGNAL_URL` 指向本机)。

```bash
# 终端 1 —— 信令服务器
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080

# 终端 2 —— 接收端(先起,等流)
pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2

# 终端 3 —— 发送端(videotestsrc 测试图)
pixi run python examples/two_cemera_sender.py
```

> 同机如果不想配 TURN,可把 `.env` 的 `TURN` 留占位符——localhost 一般能走主机候选直连。
> 跨机则几乎一定要配真实可达的 TURN。

### 4.2 跨机部署(三个角色)

推荐把 **signal-server 和 receiver 放同一台云端机(机器 B)**,sender 放边缘机(机器 A)。
设机器 B 的可达 IP 为 `B_IP`。

**机器 B(云端):跑 signal-server + receiver**
`.env` 关键项:
```bash
ROOM=demo
SIGNAL_URL=ws://127.0.0.1:18080/ws     # B 上 signal-server 是本地
TURN=turn://USER:PASS@TURN_HOST:3478?transport=udp
FLASK_HOST=0.0.0.0                      # 想从别的机器看预览就设 0.0.0.0
```
```bash
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080   # 终端1
pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2                # 终端2
```

**机器 A(边缘):跑 sender**
`.env` 关键项(`SIGNAL_URL` 必须指向 B_IP):
```bash
ROOM=demo
SIGNAL_URL=ws://B_IP:18080/ws          # ← 关键
TURN=turn://USER:PASS@TURN_HOST:3478?transport=udp
VIDEO_SOURCE=test
```
```bash
pixi run python examples/two_cemera_sender.py
```

**跨机必须打通的两件事:**
1. 机器 A → `B_IP` 的 **TCP 18080**(信令)。
2. A、B → TURN 主机的 **UDP 3478**(媒体中继)。

### 4.3 怎么判断通了

按这个顺序看日志:
- **signal-server**:`listening on :18080` → 两端连上后 `joined room=demo peer=sender-xxx` / `peer=receiver-xxx`。
- **sender**:`[sender] source=test encoder=nv/sw ...` → `[webrtc] sent offer` → `[webrtc] connection state: connected`。
- **receiver**:`[receiver] decoder=...` → `[receiver] appsink bound: cam0` → 持续刷
  `[cam0] frame (224, 224, 3) from appsink` = **端到端通了**。
- receiver 还会打印 `[flask] serving on http://HOST:5082 ...`。

---

## 5. 用真实相机(v4l2)

1. 先看有哪些设备和支持的格式:
   ```bash
   ls /dev/video*
   v4l2-ctl --list-devices                 # 若装了 v4l-utils
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```
2. 在(sender 机器的)`.env` 里:
   ```bash
   VIDEO_SOURCE=v4l2
   CAMERAS=mid=/dev/video0@30,left=/dev/video4@25
   WIDTH=640
   HEIGHT=480
   ```
3. 启动 sender:`pixi run python examples/two_cemera_sender.py`

**注意点:**
- 流的名字(`mid`/`left`)会作为 msid 传到接收端,接收端用它当推理缓冲的 key,也用于
  HTTP 预览的 `?cam=` 选择。
- example 的 v4l2 管线请求的是 `format=YUY2`。如果你的相机只支持 MJPG/其它格式,这条会失败;
  需要相应改 `examples/two_cemera_sender.py` 里 `v4l2_source()` 的 caps(把 `format=YUY2`
  换成相机支持的,或插 `jpegdec`)。
- 分辨率/帧率必须是相机真实支持的组合(用上面的 `--list-formats-ext` 确认)。

---

## 6. HTTP 实时看画面

接收端会在后台起一个 HTTP 服务(`FLASK_HOST:FLASK_PORT`,默认 `127.0.0.1:5082`),提供三个接口:

| 接口 | 作用 |
|---|---|
| `GET /api/v1/preview` | **MJPEG 实时视频流**,浏览器直接打开就能看实时画面 |
| `GET /api/v1/preview.jpg` | 最新一帧的单张 JPEG(适合脚本抓图/手动刷新) |
| `GET /api/v1/infer-buffer/base64` | 原始推理缓冲(base64,给同机进程取数据用,不是图片) |
| `GET /api/v1/healthz` | 健康检查 |

**实时看画面:** 浏览器打开
```
http://<接收端IP>:5082/api/v1/preview
```
多路时选某一路 / 限制帧率:
```
http://<接收端IP>:5082/api/v1/preview?cam=cam0&fps=15
```

**要从别的机器看:** 接收端启动前在 `.env` 设 `FLASK_HOST=0.0.0.0`(默认 `127.0.0.1` 只能本机看),
并放行该端口。

**安全提醒:** 预览和缓冲接口**没有鉴权**,`FLASK_HOST=0.0.0.0` 会把画面暴露到该网段所有人。
只在可信内网这么做,公网请加反向代理/鉴权。

> 原理:`/preview` 从 live 推理缓冲里取最新帧(HWC uint8 RGB tensor),在锁内拷到 CPU、
> 编成 JPEG 返回。`/api/v1/infer-buffer/base64` 走的是 CUDA-IPC 序列化,**跨机还原不出图**
> (句柄只在同机同 GPU 有效),所以远程"看画面"请用 `/preview`。

---

## 7. 没有 GPU(纯 CPU 跑)

整条链路都能在没有 NVENC/NVDEC 的机器上跑:

- **发送端**:`.env` 设 `ENCODER=sw`(或靠 `auto` 自动探测),用 `x264enc` 软编。
- **接收端**:`.env` 设 `DECODER=sw`(或 `auto`),用 `avdec_h264` 软解。
- **推理缓冲**:无 CUDA 时自动用 CPU tensor(功能正常,只是这段不再是 CUDA-IPC 零拷贝)。

```bash
# 无 GPU 机器的 .env
ENCODER=sw
DECODER=sw
```

> 提示:某些机器装了 `nvh264enc` 插件但运行时 NVENC session 打不开。`auto` 通常能正确探测
> (开不了会注册失败、自动退到软编);若遇到 auto 误判,显式设 `ENCODER=sw` 最稳。

---

## 8. 故障排查

| 现象 | 多半原因 |
|---|---|
| signal-server 没有 `joined` 日志 | `SIGNAL_URL` 不对 / 服务器没起 / 跨机端口 18080 不通(`nc -vz B_IP 18080`) |
| sender 一直 `connection state: connecting`/`failed` | ICE 没打通,几乎都是 **TURN**:地址/凭据错、或 UDP 3478 被挡 |
| sender 报 `nvh264enc`/`cudaupload` not found | 这台没 NVIDIA GStreamer 插件 → 设 `ENCODER=sw` |
| receiver 连上但没有 `frame ... from appsink` | 流没过来或解码失败,先查 TURN/网络;可设 `GST_DEBUG=4` 看细节 |
| 浏览器打开 `/preview` 404 / no frame | 还没收到帧(等 sender 连上),或 `?cam=` 名字不存在 |
| 远程打不开预览 | 接收端 `FLASK_HOST` 还是 `127.0.0.1`,改成 `0.0.0.0` 并放行端口 |
| v4l2 启动失败 | 相机不支持请求的 `WIDTH/HEIGHT/fps/YUY2` 组合,用 `v4l2-ctl --list-formats-ext` 核对 |

调试管线细节:任意一端加 `GST_DEBUG=4`(或更高)再启动。

---

## 9. 附:命令速查

```bash
# 信令服务器
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080

# 接收端(N 路,监听所有网卡的 5082 以便远程看预览)
FLASK_HOST=0.0.0.0 pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2

# 发送端(测试图 / 真实相机由 .env 的 VIDEO_SOURCE 决定)
pixi run python examples/two_cemera_sender.py

# 浏览器实时看画面
#   http://<接收端IP>:5082/api/v1/preview
```

更多设计/架构见 [`design.md`](design.md) 与 [`docs/`](docs/)。
