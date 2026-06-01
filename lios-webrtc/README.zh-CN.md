<div align="center">
  <img alt="LiOS" src="logo/LiOS_logo.png" height="120">

  <h1>lios-webrtc</h1>

  <p><em>面向机器人推理的低时延 · GPU 加速端云图传</em></p>

  <p>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <a href="https://github.com/cmriat/LiOS/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/cmriat/LiOS/actions/workflows/ci.yml/badge.svg"></a>
    <img alt="GStreamer" src="https://img.shields.io/badge/GStreamer-1.26-orange">
    <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-261230.svg"></a>
  </p>

  <p><a href="README.md">English</a> · <strong>简体中文</strong></p>
</div>

---

`lios-webrtc` 是 [LiOS](../README.md) 具身智能基础设施栈的**图传组件**——一条为"机器人推理"而生的**低时延视觉数据通路**。端侧多路相机视觉经 **WebRTC / GStreamer** 稳定进入云端，支撑在线推理、rollout 记录与运行复盘。

> **与面向"人观看"的通用实时音视频框架（会议 / 直播 / 协作 SDK）不同：通用框架的终点是把画面解码出来交给应用层或屏幕；LiOS 图传的终点，是把画面直接送到 GPU 上的模型（VLA / Policy）手里。**

**为什么能做到低时延？** LiOS 图传围绕"让画面以尽量短的路径进入云端模型"设计：

- **GPU 硬件编解码通路**——边缘相机 → NVENC 硬件编码 → 加密 WebRTC 传输 → 云端 NVDEC 硬件解码。编码、解码在 GPU 上完成（NVENC/NVDEC），色彩转换也可留在 GPU 侧（`nvvideoconvert`），数据面不走软件编解码。
- **零拷贝推理缓冲**——解码后的帧以 CUDA tensor 形式经 `InferenceBufferV2` 暴露给下游；它通过 CUDA-IPC 句柄在多进程间共享同一块显存，模型、观测、记录等多个消费者**映射同一块设备内存、无需额外拷贝**。
- **近网中继部署**——中继节点贴近云端推理服务，适配实验室内网、企业网络、云厂商 VPC 等环境，在保证接入可达的同时减少公网绕行。
- 由此把"本地相机 → 云端 CUDA 显存"的端到端单向延迟压到 **约 30 ms 级**（其中网络约 24 ms），较通用中继 / 公网 Cloud 方案显著更低，并支持多路相机并发上云。

组件架构与各模块职责见 [`design.md`](design.md)。

---

## 架构

![LiOS 架构](docs/gst-report/arch_imgtx.png)

红框节点运行在 GPU 上——NVENC 编码、NVDEC 解码，以及保存 CUDA tensor 的推理缓冲（经 CUDA-IPC 跨进程零拷贝共享）。详见 [`design.md`](design.md)。

---

## 快速上手

需要 NVIDIA 驱动与 GStreamer 插件（`nvh264enc` / `nvh264dec` / `cudaupload`）。环境用 [pixi](https://pixi.sh) 管理。

```bash
pixi install

# 0) 配置：复制并填入 ROOM / SIGNAL_URL / STUN / TURN
cp .env.example .env   # example 会自动加载 .env；默认配置无需真实相机

# 1) 启动信令服务（Go）
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080

# 2) 启动发送端（边缘 GPU）。默认 VIDEO_SOURCE=test 用 videotestsrc，无需相机。
pixi run python examples/two_cemera_sender.py

# 3) 启动接收端（云端 GPU：解码 → CUDA → InferenceBufferV2）
pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2

# 4) 浏览器看实时预览（接收端自动提供）
#    http://127.0.0.1:5082/api/v1/preview          （MJPEG 流）
#    http://127.0.0.1:5082/api/v1/preview?cam=cam0&fps=15
#    想从别的机器看，第 3 步前设 FLASK_HOST=0.0.0.0
```

配置从 `.env`（或环境变量，环境变量优先）读取：`ROOM`、`SIGNAL_URL`、`STUN`、`TURN`，以及
`VIDEO_SOURCE`（`test` | `v4l2`）和 `CAMERAS`。用真实相机时设 `VIDEO_SOURCE=v4l2`、
`CAMERAS=mid=/dev/video0@30,left=/dev/video4@25`。设 `GST_DEBUG=4` 调试管线。

完整教程——跨机部署、coturn/TURN 配置、真实相机、实时预览、故障排查——见
**[USER_GUIDE.zh-CN.md](USER_GUIDE.zh-CN.md)** · [English](USER_GUIDE.md)。

---

## 性能基准

所有对比都使用**完全相同的内容**（两侧编码同一段合成片段）、**对齐的码率**，并在**接收端逐帧计数**（直接统计解码帧，而非 `fpsdisplaysink` 的速率信号——它对该流会高报）。两套方案都使用 **NVENC/NVDEC 硬件**编解码；下面的差异来自架构与部署，而非硬件 vs 软件。

### 延迟（部署拓扑）

![单向延迟](docs/gst-report/latency_compare.png)

相机 → 云端 CUDA 显存的单向延迟（双向 NTP 方法，已抵消时钟偏移）。这里对比**各自的默认部署**：LiOS 自建一个可部署在机器人 / 云端推理近端的**边缘中继**（约 28 ms）；LiveKit 托管 **Cloud** 默认经其 PoP 路由（约 242 ms）。这里的优势在于可控制中继位置——LiveKit 从不使用 LiOS 的中继。

### 吞吐

![吞吐：实际 vs 目标](docs/gst-report/throughput_samecontent.png)

| 条件 | LiOS (NVENC/NVDEC) | LiveKit (libwebrtc) |
|---|---|---|
| 同片段 · 10 Mbps · 本机 | **≥ 1200 fps**（未饱和） | **≈ 600–700 fps**（已饱和；≥ 1000 fps 时订阅端流中断） |
| 跨机 · 2 Mbps · 各自默认部署 | **≈ 1687 fps**（边缘中继，稳定） | **≈ 330 fps**（Cloud，波动大 246–334） |

- LiOS 的实际帧率在测量上限（1200 fps）内始终跟随目标、不饱和；其管线天花板用 `videotestsrc` 单独测得约 **1685 fps**。
- LiveKit 的单流实时管线在 **700 fps** 附近饱和；≥ 1000 fps 时订阅端流中断。
- 有时被归给这类 SDK 的"25 fps"是**配置问题**（发布端未设 `max_framerate`，默认约 30 fps），而非硬件限制——设好帧率后 libwebrtc 可稳定 200–700 fps。

### 投递抖动

![帧间间隔小提琴图](docs/gst-report/throughput_jitter_violin.png)

在同为 **400 fps** 平均吞吐下，帧间到达间隔：LiOS **σ = 0.08 ms**（p95 2.64），LiveKit **σ = 0.66 ms**（p95 3.48，max 5.4）——抖动约 **8 倍**。对 VLA / 实时控制而言，投递间隔的方差比平均帧率更重要。

---

## 目录结构

| 路径 | 说明 |
|---|---|
| `src/gst_webrtc/sender/` | `WebRTCSender`（webrtcbin sendonly，gst-launch 风格动态加源） |
| `src/gst_webrtc/receiver/` | `WebRTCReceiver`（webrtcbin recvonly，RTP-sink 描述串） |
| `src/gst_webrtc/gpu_sink/` | `GpuFrameSink`：appsink → 线程安全帧队列 |
| `src/gst_webrtc/inference_buffer_v2.py` | `InferenceBufferV2`：CUDA-IPC 零拷贝推理缓冲（posix 共享内存） |
| `src/gst_webrtc/ws_signal/` | WebSocket 信令客户端 |
| `src/gst_webrtc/services/` | 状态同步 / 控制 API（Flask + WebSocket JSON） |
| `signal-server/` | Go WebRTC 信令服务（Cobra CLI） |
| `examples/` | 可运行的双相机发送 / 接收（含推理缓冲） |
| `benchmark/` | 基准：`make_figures.py`（所有图表）、`throughput/`（gst）、`livekit/`（LiveKit）、`rtp_latency/`（RTP 延迟探针）——见 [`benchmark/README.md`](benchmark/README.md) |
| `docs/` | 实验报告与图表 |

---

## 文档

- [`design.md`](design.md) — 组件架构与各模块职责
- [`docs/gst-report/`](docs/gst-report/) — 图传性能实验报告（延迟 + 吞吐）
- [`AGENTS.md`](AGENTS.md) — 开发约定

---

## 许可证

本项目以 [Apache License 2.0](LICENSE) 开源。

---
