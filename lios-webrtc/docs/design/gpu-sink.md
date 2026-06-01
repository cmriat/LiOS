# gst_webrtc.gpu_sink — GPU Frame Sink 设计文档

目标：在接收端（WebRTC/本地 RTP 解码链路）把 GPU 解码/处理出来的视频帧，稳定、低延迟地桥接到 Python 侧的 NumPy 数组，用于后续 CV/AI 处理。组件以 GStreamer `appsink` 为落点，提供线程安全的队列 API。

适配现有代码：
- 发送端参考 `tests/e2e/sender.py`：H264 硬编（`nvh264enc`）→ `rtph264pay` → webrtc 发送。
- 接收端参考 `tests/e2e/receiver.py`：webrtc 收到 RTP → `rtph264depay` → `h264parse` → H264 硬解（`nvh264dec`）→ `videoconvert`/`nvvideoconvert` → 显示。
- 本组件将“显示”替换为 `appsink`，并封装 NumPy 拉取接口。

为什么这样做
- `appsink` 提供稳定的回调/拉取模型，易于跨线程交付帧。
- NVIDIA `nvcodec`/`nvvideoconvert` 能把 NVMM（GPU）内存转换/下载成普通 `video/x-raw`（CPU）格式，便于 Python 映射成 NumPy。
- 回调里立刻 copy 出独立的 NumPy，避免 GStreamer buffer 生命周期问题，保证跨线程安全。

注意：NumPy 位于 CPU 内存。若后续需要 GPU 张量，可在队列消费者侧用 CuPy/PyTorch 再上载到 GPU；或未来扩展为 CUDA/GL 原生下载（见“扩展与局限”）。

---

## 组件形态与命名

模块路径：`gst_webrtc/gpu_sink/`（Python 包）；对外主要类：`GpuFrameSink`。

核心职责
- 构造“RTP → 解码 →（GPU→CPU 下载/色彩转换）→ appsink”这一段 GStreamer 描述串；或只构造“转换→appsink”的尾段，供外部拼接。
- 将 `appsink` 的帧转成 `numpy.ndarray`，塞入线程安全队列。
- 暴露拉取、清空、统计等 API。

---

## 推荐 GStreamer 管线（H264 示例）

接收端（替换 `tests/e2e/receiver.py` 的显示部分）：

```
capsfilter caps="application/x-rtp" ! rtph264depay ! h264parse !
nvh264dec !
    nvvideoconvert ! video/x-raw,format=RGBA !
    appsink name={name} emit-signals=true sync=false max-buffers=1 drop=true
```

说明
- `nvh264dec`：NVIDIA 硬解码；若缺失，回退到 `avdec_h264`。
- `nvvideoconvert`：将 NVMM/GPU 内存转换并可下行到普通 `video/x-raw`；若缺失，回退 `videoconvert`。
- `appsink`：开启 `emit-signals` 以使用 `new-sample` 回调；设置 `max-buffers=1 drop=true` 保持低延迟不阻塞上游。

兼容路径
- 无 NVIDIA 插件时：`avdec_h264 ! videoconvert ! video/x-raw,format=RGB ! appsink ...`
- GL 管线时（如经 `glupload`）：使用 `gldownload ! video/x-raw,format=RGBA ! appsink ...`。

---

## Public API 设计

命名空间：`gst_webrtc.gpu_sink`

数据结构
- `Frame`（dataclass）
  - `array: np.ndarray`  // HWC 排列，默认 `uint8`，格式由 `output_format` 决定（RGB/BGR/RGBA/GRAY8）
  - `pts_ns: int | None` // PTS（纳秒）
  - `dts_ns: int | None`
  - `seqnum: int | None`
  - `width: int`
  - `height: int`
  - `format: str`       // GStreamer negotiated format（如 RGBA）

类与方法
- `class GpuFrameSink:`
  - `__init__(self, name: str | None = None, output_format: str = "RGBA", queue_size: int = 4, drop_when_full: bool = True)`
    - 作用：配置 sink 名称、输出格式（`RGBA|RGB|BGR|GRAY8`）、内部队列容量、队列满时是否丢帧。
    - 约定：`name` 若缺省，内部生成 UUID；用于从 pipeline 里 `get_by_name()` 找到 `appsink`。

  - `rtp_h264_sink_desc(self) -> str`
    - 作用：返回一段可直接传给 `WebRTCReceiver.set_rtp_sink_desc(...)` 的完整描述串（如上“推荐管线”）。
    - 行为：优先使用 `nvh264dec + nvvideoconvert`；若运行时探测不到插件，则回退到 `avdec_h264 + videoconvert`。输出 caps 固定为 `video/x-raw,format={output_format}`。

  - `appsink_tail_desc(self) -> str`
    - 作用：只返回“颜色/内存转换 + appsink”的尾段，便于与自定义解码链拼接。
    - 示例：`"nvvideoconvert ! video/x-raw,format=RGBA ! appsink name={name} emit-signals=true sync=false max-buffers=1 drop=true"`

  - `bind(self, pipeline: Gst.Pipeline) -> None`
    - 作用：在外部把包含该 `appsink` 的 bin 加入管线后，调用此方法以获取 `appsink` 元素、设置属性并连接 `new-sample` 回调。
    - 典型调用时机：在 `WebRTCReceiver._on_pad_added()` 里完成 bin 加入与 `sync_state_with_parent()` 后立即调用。

  - `pull(self, timeout: float | None = None) -> Frame | None`
    - 作用：从内部队列取出一帧；`timeout=None` 则阻塞直到有帧。

  - `flush(self) -> int`
    - 作用：清空队列，返回丢弃的帧数。

  - `close(self) -> None`
    - 作用：断开与 `appsink` 的连接、释放内部资源。

  - `stats(self) -> dict`
    - 作用：返回简单运行指标（排队帧数、累计丢弃数、最后一帧时间戳等）。

内部方法（实现要点）
- `_on_new_sample(self, sink: Gst.Element) -> Gst.FlowReturn`
  - `sample = sink.emit("pull-sample")` → `buffer = sample.get_buffer()`
  - 从 `sample.get_caps()` 读取 `width/height/format`；`buffer.map(Gst.MapFlags.READ)` 得到 `map_info.data`。
  - 根据 `output_format` 组织 `np.ndarray`（HWC）；默认在回调里 `copy()`，以确保返回后 GStreamer 可安全复用 buffer。
  - 队列：若满且 `drop_when_full=True`，丢弃最旧/最新（实现可选其一，建议丢弃最旧保持最新帧），并更新统计。

- `_map_to_numpy(self, map_info, width, height, format) -> np.ndarray`
  - `RGBA/RGB/BGR/GRAY8` 直接 `np.frombuffer(...).reshape(h, w, C)`；
  - 若收到 `NV12/I420` 等 YUV，可临时输出原始 YUV（二维/三平面）或在 Python 端用 OpenCV 转换（可做为后续增强）。

---

## 与 WebRTCReceiver 的对接

最少侵入的方式：继续使用 `WebRTCReceiver.set_rtp_sink_desc()`，把 `GpuFrameSink` 生成的描述串交给它；随后在 `_on_pad_added()` 完成 bin 添加后，调用 `bind()` 连接回调。

示例（基于 `tests/e2e/receiver.py` 改造）：

```python
from gst_webrtc.receiver import WebRTCReceiver
from gst_webrtc.gpu_sink import GpuFrameSink

receiver = WebRTCReceiver(ROOM, "receiver", SIGNAL_URL, STUN, TURN)
sink = GpuFrameSink(output_format="RGBA", queue_size=4)
receiver.set_rtp_sink_desc(sink.rtp_h264_sink_desc())

async def main():
    # 启动后，等待 webrtc src pad 出现并完成 bin 加入
    asyncio.create_task(receiver.run())
    # 轮询直到 appsink 出现在 pipeline 里
    while True:
        el = receiver.pipe.get_by_name(sink.name)
        if el is not None:
            sink.bind(receiver.pipe)
            break
        await asyncio.sleep(0.01)

    # 消费帧
    while True:
        frame = sink.pull(timeout=1.0)
        if frame is None:
            continue
        arr = frame.array  # np.ndarray (H, W, 4) RGBA
        # ... 你的处理逻辑 ...
```

说明
- 若希望 `GpuFrameSink` 自行完成“等待并绑定”，可在实现中提供 `install_in(receiver: WebRTCReceiver)` 辅助方法；文档这里给出最通用的显式绑定方式。

---

## 与 WebRTCSender 的关系

发送端无需本组件。但若未来要把 NumPy/CV 帧注入推流（反向方向），建议另行设计 `GpuFrameSource`（`appsrc` 封装）以与 `WebRTCSender.add_video_source_desc(...)` 协作；本组件不涵盖此功能。

---

## 错误处理与降级策略

- 插件探测：运行时检查 `nvh264dec`/`nvvideoconvert` 是否可用，不可用则自动改用 `avdec_h264`/`videoconvert`；必要时在 `stats()` 里暴露“当前解码/转换后端”。
- 回调异常：`_on_new_sample()` 捕获异常后打印警告并返回 `Gst.FlowReturn.OK`（不阻塞上游），同时计入错误计数。
- 性能保护：始终保持 `appsink max-buffers=1 drop=true` 与上游 `queue leaky=downstream` 的组合，优先展示最新帧，牺牲中间帧以换取低延迟。

---

## 扩展与局限

- GPU 原生张量：可新增可选路径 `cudadownload`/`cudaconvert`（或 GL 的 `gldownload`）以获得更高效的 GPU→CPU 下载，或直接得到 CUDA 设备指针后封装为 CuPy；但 Python 侧对 NVMM/GL 映射支持有限，需谨慎评估可移植性。
- 多流：可实例化多个 `GpuFrameSink`，分别在不同 webrtc `src_%u` 上绑定，每个 sink 用不同的 `name` 与队列；需要上层应用管理路由。

---

## 与现有测试脚本的对应关系

- `tests/e2e/sender.py`：保持不变（仍然使用 `nvh264enc ! h264parse ! rtph264pay`）。
- `tests/e2e/receiver.py`：将 `videoconvert ! autovideosink` 部分替换为本文“推荐管线”的尾段（`nvvideoconvert ! ... ! appsink`），并按上节示例在 Python 中消费帧。

---

## 实现清单（方法与职责）

必须实现
- `GpuFrameSink.__init__(...)`：参数校验、生成 `name`、初始化队列与统计。
- `GpuFrameSink.rtp_h264_sink_desc()`：返回完整 sink 描述串（含回退逻辑）。
- `GpuFrameSink.appsink_tail_desc()`：返回“转换→appsink”的尾段描述串。
- `GpuFrameSink.bind(pipeline)`：按 `name` 获取 `appsink`，设置属性并连接 `new-sample` 回调。
- `GpuFrameSink.pull(timeout)`：阻塞/超时地取出一帧 `Frame`。
- `GpuFrameSink.flush()`：清空队列。
- `GpuFrameSink.close()`：断开信号、释放资源。
- `GpuFrameSink.stats()`：返回运行统计（如 `{"queued": int, "dropped": int, "backend": "nv+h264dec/nvvideoconvert|cpu"}`）。

内部/私有
- `_on_new_sample(sink)`：把 `Gst.Sample` 映射为 `Frame` 并入队；遵循丢帧策略。
- `_map_to_numpy(map_info, w, h, fmt)`：完成格式到 NumPy 的内存视图与复制。

---

## 参考

- GStreamer `nvh264dec`（nvcodec）：硬件 H.264 解码器，适用于 NVIDIA GPU。
- NVIDIA `nvvideoconvert`：在 NVMM 与普通 `video/x-raw` 间转换/下载，支持多种格式，便于交给 `appsink` 映射到 NumPy。
- GStreamer GL `gldownload`：从 GL 纹理下载为 `video/x-raw`。
- Python/Appsink → NumPy 的常见写法：`buffer.map(READ)` → `np.frombuffer(...).reshape(...)`。

> 上述能力在不同平台/驱动版本上存在差异；实现时应做插件与格式探测，并保持回退路径。
