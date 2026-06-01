# InferenceBufferV2 — 自包含、全 base64 的推理缓冲区

本实现在 `src/gst_webrtc/inference_buffer_v2.py` 中，目标是：

- 自包含（pickle 中自包含）：序列化结果解包时不需要导入 stub；接收方只需有 `torch` 和 Python 标准库即可还原为普通 `dict` 与 `torch.Tensor`。
- CUDA IPC：对 CUDA Tensor 使用 Torch 的 ForkingPickler 规约，传递的是 CUDA IPC 句柄，跨进程零拷贝共享显存。
- 携带元信息：在包内嵌入 `meta` 字段（例如 `frame_id`、`timestamp`、`model`、`fps` 等），并带有 `header`（`__ifbuf__` 版本与 `created_ns`）。
- 全 base64：输出为 ASCII 字符串，易于放入 JSON、WebSocket、gRPC、消息队列等载体。

注：CUDA IPC 需要导出方在接收方使用期间保持源 Tensor 存活。否则行为未定义。两端都必须可用同一块 GPU（或可见的 NVLink/同机）并有 `torch`。

## 快速上手

发送端（构造并打包）：

```python
import base64, pickle
import torch
from gst_webrtc.inference_buffer_v2 import InferenceBufferV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.zeros((3, 480, 640), device=device, dtype=torch.float32)  # HWC/RGB 仅作示例

buf = InferenceBufferV2(
    images={"rgb": x},
    states={},
    prev_actions={},
    meta={"frame_id": 42, "model": "resnet50", "ts_ns": 1234567890},
)

b64 = buf.pack_base64()  # 或 buf.to_base64()
# 现在可以把 b64 放到 JSON 或直接通过 socket 发送
```

接收端 A（完全不导入本模块，直接用 stdlib + torch 复原为 `dict`）：

```python
import base64, pickle

# s 是从网络收到的 base64 字符串
payload = pickle.loads(base64.b64decode(s))

assert "images" in payload and "meta" in payload
rgb = payload["images"]["rgb"]  # torch.Tensor（CUDA 情况下是同一显存的映射）
print(payload["meta"])  # {"frame_id": 42, ...}
```

接收端 B（若你在项目中可导入本模块，也可包装回 dataclass）：

```python
from gst_webrtc.inference_buffer_v2 import InferenceBufferV2

buf2 = InferenceBufferV2.from_base64(s)
rgb = buf2.images["rgb"]
print(buf2.meta)
```

## API 速览

- `InferenceBufferV2(images, states, prev_actions, meta)`：数据容器。
- `InferenceBufferV2.pack_base64()` / `to_base64()`：返回自包含 base64 字符串。
- `InferenceBufferV2.pack_bytes()` / `to_bytes()`：返回二进制 pickle 字节串。
- `InferenceBufferV2.from_base64(s)` / `from_bytes(b)`：从 base64/bytes 恢复 dataclass。
- `InferenceBufferV2.to_plain_payload()`：返回可 pickle 的普通 `dict`（便于调试/单测）。
- `pack_base64(images=..., states=..., prev_actions=..., meta=...)`：模块级函数，直接打包为 base64；接收端可不依赖本模块。
- `unpack_base64(s)`：模块级函数，解包为普通 `dict`（无 dataclass）。

返回的普通 `dict` 结构如下：

```python
{
  "header": {"__ifbuf__": 1, "created_ns": 1730860000000000000},
  "meta": {"frame_id": 42, "model": "resnet50", "ts_ns": 1234567890},
  "images": {"rgb": <torch.Tensor>},
  "states": {"...": <torch.Tensor>},
  "prev_actions": {"...": <torch.Tensor>},
}
```

## 生命周期与安全注意事项

- CUDA IPC 生命周期：导出方必须在所有消费者使用期间维持源 `torch.Tensor` 存活并在同一计算环境/设备可见范围内。删除原 Tensor 或释放对应 CUDA 上下文会导致接收方访问失效。
- 线程/进程：推荐使用 `torch.multiprocessing` 的 `spawn` 启动方式，以避免 fork 后的 CUDA 上下文问题。
- 序列化安全：仅在信任边界内反序列化（`pickle.loads` 具有代码执行风险）。跨信任边界请加签名/鉴权或改用安全格式。

## 端到端微测脚本（可选）

```python
import torch, base64, pickle
from gst_webrtc.inference_buffer_v2 import pack_base64, unpack_base64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src = torch.zeros((2, 3), device=device, dtype=torch.float32)
b64 = pack_base64(images={"x": src}, meta={"note": "demo"})

plain = unpack_base64(b64)          # 无需导入 dataclass
dst = plain["images"]["x"]
dst.fill_(1.0)
torch.cuda.synchronize() if dst.is_cuda else None
assert torch.allclose(src, torch.ones_like(src))  # 若为 CUDA，零拷贝共享
```

---

文件：`src/gst_webrtc/inference_buffer_v2.py`。
版本：`__ifbuf__ = 1`（若变更结构将会递增）。

