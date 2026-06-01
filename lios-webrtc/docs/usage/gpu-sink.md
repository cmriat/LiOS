# GPU Frame Sink 使用说明（gst_webrtc.gpu_sink）

本组件提供一个最小可用的 GPU → CPU 帧下行与队列封装：把接收到的 H264 RTP 流，经硬/软解码与颜色转换后，落到 `appsink`，并在 Python 侧以线程安全队列的方式提供 `numpy.ndarray`。

实现遵循 docs/design/gpu-sink.md，目标是“保证基本流程跑通”，便于后续在 CV/AI 里消费帧。

## 快速开始

前置：
- 按仓库根的 AGENTS.md 完成环境：`pixi install && pixi shell`。
- 启动信令：`cd signal-server && go build -o signalsrv . && ./signalsrv serve --addr :18080`。
- 启动发送端（参考 `tests/e2e/sender.py`）。

在接收端中使用 `GpuFrameSink`：

```python
import asyncio, os
import numpy as np
from gst_webrtc import init_gst
from gst_webrtc.receiver import WebRTCReceiver
from gst_webrtc.gpu_sink import GpuFrameSink

ROOM = os.environ.get("ROOM", "demo")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.example.com")

init_gst()

receiver = WebRTCReceiver(ROOM, "receiver", SIGNAL_URL, STUN)
sink = GpuFrameSink(output_format="RGBA", queue_size=4)
receiver.set_rtp_sink_desc(sink.rtp_h264_sink_desc())

async def main():
    asyncio.create_task(receiver.run())
    # 轮询直到 appsink 出现在 pipeline 中，然后绑定回调
    while True:
        el = receiver.pipe.get_by_name(sink.name)
        if el is not None:
            sink.bind(receiver.pipe)
            break
        await asyncio.sleep(0.01)

    # 消费帧：frame.array → numpy.ndarray[H, W, C]
    while True:
        frame = sink.pull(timeout=1.0)
        if frame is None:
            continue
        arr = frame.array  # RGBA(uint8)
        # 示例：读取中心像素
        h, w = arr.shape[:2]
        px = arr[h//2, w//2]
        print("center RGBA:", px, "PTS(ns):", frame.pts_ns)

asyncio.run(main())
```

保存为 `examples/recv_gpu_sink.py` 后，可用：

```
pixi run python examples/recv_gpu_sink.py
```

> 注意：上述示例中 `sink.bind(...)` 的时机很关键，需等待 `receiver` 在 `_on_pad_added` 里把包含 `appsink` 的 bin 加入到 pipeline 后再绑定。

## API 概览

- `GpuFrameSink(output_format="RGBA", queue_size=4, drop_when_full=True)`: 输出格式可选 `RGBA|RGB|BGR|GRAY8`。
- `rtp_h264_sink_desc() -> str`: 返回完整 H264 RTP 解码到 `appsink` 的描述串；优先 `nvh264dec + nvvideoconvert`，自动探测失败则回退到 `avdec_h264 + videoconvert`。
- `appsink_tail_desc() -> str`: 仅返回“颜色/内存转换 + appsink”的尾段，便于自定义拼接。
- `bind(pipeline)`: 根据 `name` 找到 `appsink` 并连接 `new-sample` 回调。
- `pull(timeout=None) -> Frame | None`: 从内部队列取出一帧；超时返回 `None`。
- `flush() -> int`: 清空队列并返回丢弃的帧数。
- `close()`: 断开回调、释放资源。
- `stats() -> dict`: 返回运行指标（队列长度、累计丢弃/错误、最后 PTS/当前后端等）。

`Frame` 字段：`array(np.ndarray), pts_ns, dts_ns, seqnum, width, height, format`。

## 管线说明与降级

- NVIDIA 插件存在时：`nvh264dec ! nvvideoconvert ! video/x-raw,format=RGBA ! appsink`。
- 否则回退：`avdec_h264 ! videoconvert ! video/x-raw,format=RGBA ! appsink`。
- 始终开启低延迟选项：`appsink emit-signals=true sync=false max-buffers=1 drop=true` 与上游 `queue leaky=downstream`。

## 实现备注（当前最小版本）

- 回调中总是 `copy()` 一份独立的 `numpy` 数组，确保线程安全和避免 buffer 生命周期问题；`copy_frame=False` 目前按兼容处理，依然会拷贝。
- 若底层帧行对齐导致行跨度（stride）大于 `width*channels`，实现会以 `height×(stride/channels)` 的形状读入并在列上裁切到 `width`，以保证输出数组形状与语义正确。

## 故障排查

- 报错 `webrtcbin not available`：检查 `pixi install` 是否成功，或系统 GStreamer 安装。
- 解码失败或黑屏：`nvh264dec/nvvideoconvert` 可能不可用，确认 NVIDIA 驱动与相关插件；或让其自动回退到 CPU 解码（已内置）。
- 高延迟：确认 `GST_DEBUG` 设置，检查是否有上游元素阻塞；必要时提升到 `GST_DEBUG=4` 查看日志。

