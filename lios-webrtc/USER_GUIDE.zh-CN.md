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
pixi run python examples/two_camera_receiver_inferbuf.py --streams 2

# 终端 3 —— 发送端(videotestsrc 测试图)
pixi run python examples/two_camera_sender.py
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
pixi run python examples/two_camera_receiver_inferbuf.py --streams 2                # 终端2
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
pixi run python examples/two_camera_sender.py
```

**跨机必须打通的两件事:**
1. 机器 A → `B_IP` 的 **TCP 18080**(信令)。
2. A、B → TURN 主机的 **UDP 3478**(媒体中继)。

**18080 不能直连时(只有 SSH 通道,如 Coder workspace):用 SSH 端口转发**

不改 `.env`(两端仍写 `127.0.0.1:18080`),在 **A(sender 机器)** 上把本地 18080
隧道到 B 的 signal-server:
```bash
ssh -N -L 18080:127.0.0.1:18080 <RECEIVER_SSH_HOST>
# <RECEIVER_SSH_HOST> 换成你 SSH 进 B(receiver 机器)用的名字,如 user@1.2.3.4 或 ~/.ssh/config 里的别名
```
保持这个终端开着;隧道活着时 A 上 `ws://127.0.0.1:18080/ws` 就直达 B 的 signal-server。
**只需转发信令这一个端口**——媒体走 TURN(`?transport=udp` 那台),不经隧道。
顺手把预览端口也带上(`-L 5082:127.0.0.1:5082`)就能在本地浏览器看 B 的实时画面。

### 4.3 怎么判断通了

按这个顺序看日志:
- **signal-server**:`listening on :18080` → 两端连上后 `joined room=demo peer=sender-xxx` / `peer=receiver-xxx`。
- **sender**:`[sender] source=test encoder=nv/sw ...` → `[webrtc] sent offer` → `[webrtc] connection state: connected`。
- **receiver**:`[receiver] decoder=...` → `[receiver] appsink bound: cam0` → 持续刷
  `[cam0] frame (224, 224, 3) from appsink` = **端到端通了**。
- receiver 还会打印 `[flask] serving on http://HOST:5082 ...`。

### 4.4 配 coturn(跨机几乎必需)

同机 localhost 不用 TURN(走 host 候选直连)。**一旦跨机/跨 NAT**,两端 host 候选互相
不可达,就得靠一台 TURN 中继转发媒体。LiOS 的核心做法是把这台 coturn 部署在**靠近云端
GPU(机器 B)**的位置,媒体绕行最短。

代码只吃**一条** TURN URI(`webrtcbin` 的 `turn-server` 属性),用长期凭据
(username/password),所以 coturn 配一个用户就够。

**1) 装(Ubuntu/Debian):**
```bash
sudo apt-get install -y coturn
```

**2) 写 `/etc/turnserver.conf`(最小可用):**
```ini
listening-port=3478
fingerprint
lt-cred-mech                       # 长期凭据机制,和下面 user= 配套
user=lios:S3cret-change-me         # 用户名:密码,要和 .env 里的 TURN 一致
realm=lios
external-ip=B_PUBLIC_IP            # 中继机在 NAT/云后面时必填:写它的公网/可达 IP
min-port=49152                     # 中继端口范围,防火墙要一起放行
max-port=65535
no-tls                             # 不用 TLS(turns://);只跑明文 turn://
no-dtls
no-cli
```

**3) 起服务:**
```bash
# 方式 A:systemd(先让默认配置允许启动)
echo 'TURNSERVER_ENABLED=1' | sudo tee /etc/default/coturn
sudo systemctl enable --now coturn

# 方式 B:前台直接跑(调试时看日志最直观)
turnserver -c /etc/turnserver.conf -v
```

**4) 防火墙放行(关键,漏了就是 sender 一直 `connecting`):**
- `UDP/TCP 3478` —— TURN 控制端口
- `UDP 49152-65535` —— 中继媒体端口范围(和上面 `min/max-port` 对齐)

**5) 两端 `.env` 填同一条(与 `user=` 凭据、中继机 IP 一致):**
```bash
TURN=turn://lios:S3cret-change-me@B_PUBLIC_IP:3478?transport=udp
```

**6) 验证 TURN 本身通不通(不依赖整条图传链路):**
```bash
# coturn 自带压测客户端,能拿到 relay 地址就说明凭据+端口都对
turnutils_uclient -v -u lios -w S3cret-change-me B_PUBLIC_IP
# 或浏览器开 Trickle ICE 页,填这条 turn:// 看能否收到 typ relay 候选
```
sender 日志走到 `connection state: connected`、receiver 持续刷 `frame ... from appsink`
就是连 TURN 在内整条链路通了。

> 注:`transport=udp` 指 sender/receiver **到 TURN** 这一跳用 UDP;coturn 默认 3478 同时
> 听 UDP/TCP。若客户端网络只放行 TCP,把 URI 改成 `?transport=tcp`(并确保 3478/TCP 放行)。

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
3. 启动 sender:`pixi run python examples/two_camera_sender.py`

**注意点:**
- 流的名字(`mid`/`left`)会作为 msid 传到接收端,接收端用它当推理缓冲的 key,也用于
  HTTP 预览的 `?cam=` 选择。
- example 的 v4l2 管线请求的是 `format=YUY2`。如果你的相机只支持 MJPG/其它格式,这条会失败;
  需要相应改 `examples/two_camera_sender.py` 里 `v4l2_source()` 的 caps(把 `format=YUY2`
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
并放行该端口。**只有 SSH 通道时**(如 Coder workspace),保持默认 `127.0.0.1` 并在本地转发即可:
```bash
ssh -N -L 5082:127.0.0.1:5082 <RECEIVER_SSH_HOST>   # 本地开 http://127.0.0.1:5082/api/v1/preview
```

**打不开?先用 curl 绕开代理自查**(把"服务/链路有没有数据"和"浏览器配错"分开):
```bash
curl -sS --noproxy '*' -o /tmp/p.jpg -w "HTTP %{http_code} %{size_download}B\n" \
  http://127.0.0.1:5082/api/v1/preview.jpg
```
- 出 `HTTP 200` 且有几 KB → 服务正常,问题在**浏览器代理**(把 `localhost,127.0.0.1` 加进不走代理白名单)。
- `HTTP 404` / `0B` → 服务起了但**还没帧**,回 receiver 看有没有刷 `frame ... from appsink`。
- `Couldn't connect` → Flask 没起或端口/隧道不对,回 receiver 找 `[flask] serving on ...`。

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
| sender 握手报 `InvalidMessage` / `did not receive a valid HTTP response` | **本机开了代理**:`websockets`(v15+)会把 `ws://127.0.0.1` 也按 `ALL_PROXY`/`HTTPS_PROXY` 走代理而握手失败。`export no_proxy=localhost,127.0.0.1`(或临时关代理)再起 |
| sender 一直 `connection state: connecting`/`failed` | ICE 没打通,几乎都是 **TURN**:地址/凭据错、或 UDP 3478 被挡 |
| sender 报 `nvh264enc`/`cudaupload` not found | 这台没 NVIDIA GStreamer 插件 → 设 `ENCODER=sw` |
| receiver 连上但没有 `frame ... from appsink` | 流没过来或解码失败,先查 TURN/网络;可设 `GST_DEBUG=4` 看细节 |
| 浏览器打开 `/preview` 404 / no frame | 还没收到帧(等 sender 连上),或 `?cam=` 名字不存在 |
| `curl` 能出图但浏览器打不开预览 | **浏览器代理**把 `127.0.0.1` 转发走了,把 `localhost,127.0.0.1` 加进浏览器"不走代理"白名单 |
| 远程打不开预览 | 接收端 `FLASK_HOST` 还是 `127.0.0.1`,改成 `0.0.0.0` 并放行端口;或用 SSH 转发 5082(见下) |
| v4l2 启动失败 | 相机不支持请求的 `WIDTH/HEIGHT/fps/YUY2` 组合,用 `v4l2-ctl --list-formats-ext` 核对 |

调试管线细节:任意一端加 `GST_DEBUG=4`(或更高)再启动。

---

## 9. 附:命令速查

```bash
# 信令服务器
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080

# 接收端(N 路,监听所有网卡的 5082 以便远程看预览)
FLASK_HOST=0.0.0.0 pixi run python examples/two_camera_receiver_inferbuf.py --streams 2

# 发送端(测试图 / 真实相机由 .env 的 VIDEO_SOURCE 决定)
pixi run python examples/two_camera_sender.py

# 浏览器实时看画面
#   http://<接收端IP>:5082/api/v1/preview
```

更多设计/架构见 [`design.md`](design.md) 与 [`docs/`](docs/)。
