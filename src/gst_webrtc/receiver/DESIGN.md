# WebRTC Receiver 设计（精简版）

目标：在 `gst_webrtc/receiver/` 下实现一个“简洁可组合”的 WebRTC 接收端，
像 `gst_webrtc/sender/core.py` 一样通过「gst‑launch 风格的描述串」添加/拼装
webrtc 输出的 RTP 到你自定义的下游链路；删除解码器探测与冗长的动态拼接逻辑，
仅保留必要的信令响应（O/A、候选）与最小化的恢复语义（被动处理 ICE 重启）。

## 架构与职责（精简）

- 类 `WebRTCReceiver`：
  - 管理 `Pipeline` 与 `webrtcbin`（recvonly）。
  - 提供 `set_rtp_sink_desc(desc: str)`：传入一段以 `application/x-rtp` 为输入的
    GStreamer 描述串（例如 H264/VP8 的 depay/parse/dec/显示链），内部用
    `Gst.parse_bin_from_description` 构建 `Bin` 并加入 `Pipeline`。
  - `webrtcbin` 触发 `pad-added` 时，将其 `src_%u` pad 连接到上述 `Bin` 的 `sink`。
  - 信令：被动处理 `offer`、发送 `answer`、收发候选；不主动发 offer，不实现复杂状态机。
  - 恢复（最小）：发送端断联后触发 ICE 重启时，接收端在收到新 offer 后按 O/A 流程答复即可。

## 关键时序（断联→ICE 重启→接收端启动）

1) 发送端检测 `ice-connection-state` 异常，根据 `sender/core.py` 触发 ICE 重启
   并发送带新 `ice-ufrag/pwd` 的 offer。

2) 接收端启动并 `join/ready` 后收到新 offer：
   - `set-remote-description(offer)`；
   - `create-answer()` → `set-local-description(answer)`；
   - 通过信令发送 answer；
   - flush 之前积压的本地候选（在未知 `remote_id` 时产生）。

3) 候选处理（最小）：
   - 远端候选在未 `set-remote` 前缓存，之后统一 `add-ice-candidate`。
   - 本地候选在未获 `remote_id` 前缓存，答复后 flush。

## 解码链（通过描述串传入）

- 通过 `set_rtp_sink_desc(desc)` 传入完整链路（从 `application/x-rtp` 到最终 sink）。
- 示例：
  - H264: `"capsfilter caps=application/x-rtp ! rtph264depay ! h264parse config-interval=-1 ! avdec_h264 ! videoconvert ! autovideosink"`
  - VP8:  `"capsfilter caps=application/x-rtp ! rtpvp8depay ! vp8dec ! videoconvert ! autovideosink"`

## 与 sender/core.py 的协同

- 发送端主导“完美协商 + ICE 重启”；接收端不主动发 offer。
- 接收端只需：正确响应 `offer` 与候选，`pad-added` 时把 webrtc 的 `src` 链接到描述串 Bin 的 `sink`。

## 失败与恢复小结（最小）

- 接收端只需“被动”处理 ICE 重启带来的新 offer 并答复即可。
- 不额外实现信令重连与复杂状态机；如需更强健，可在后续迭代添加。

## 可测试要点

- 启动发送端（H264 或 VP8）→ 启动接收端并以相应描述串添加 RTP sink：应正常显示。
- 触发发送端 ICE 重启：接收端收到新 offer 后自动答复并恢复媒体。
