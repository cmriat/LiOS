"""
Receive two WebRTC H264 streams and write frames into
`InferenceBufferV2.images` keyed by each stream's msid (e.g., "cam0", "cam1").

Usage
-----
- Configure once: `cp .env.example .env` and fill in ROOM / SIGNAL_URL / STUN / TURN.
  Config is auto-loaded via gst_webrtc.load_env() (env vars still win).
- Start signaling server (see project docs).
- Start the sender:
    pixi run python examples/two_cemera_sender.py
- Start this receiver:
    pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2

Notes
-----
- Appsink elements are renamed to the incoming pad's `msid` by WebRTCReceiver,
  so we use those names as `images` keys in the inference buffer.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Dict, Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstApp", "1.0")
import numpy as np
import torch
from gi.repository import Gst, GstApp  # type: ignore
from PIL import Image

from gst_webrtc import init_gst, load_env
from gst_webrtc.inference_buffer_v2 import InferenceBufferV2
from gst_webrtc.receiver import WebRTCReceiver
from gst_webrtc.services.flask_api import APIServer, InferenceBufferProvider


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL.

    Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image

def relax_rtp_tolerance(receiver: WebRTCReceiver):
    webrtc = receiver.webrtc
    rtpbin = None

    # 方式A：直接按名字取（webrtcbin 是个 Bin，内部 child 名就叫 "rtpbin"）
    try:
        # 有的GI封装提供 child proxy：
        if hasattr(webrtc, "get_child_by_name"):
            rtpbin = webrtc.get_child_by_name("rtpbin")
    except Exception:
        pass

    # 方式B：从整个 pipeline 里递归找名为 rtpbin 的元素（通用兜底）
    if rtpbin is None:
        it = receiver.pipe.iterate_recurse()
        try:
            while True:
                res, elem = it.next()
                if res == Gst.IteratorResult.OK and elem.get_name() == "rtpbin":
                    rtpbin = elem; break
                if res != Gst.IteratorResult.OK:
                    break
        except Exception:
            pass
        finally:
            it = None

    if not rtpbin:
        print("[tune] rtpbin not found"); return

    # 2.1 新的 SSRC 一验证就把 probation 放到 1，并放宽乱序/掉包时间
    session = rtpbin.emit("get-internal-session", 0)  # WebRTC/BUNDLE 通常用 session 0
    if session:
        def on_validated(_session, src, *a):
            try:
                src.set_property("probation", 1)             # 连续1个包即可通过
                src.set_property("max-misorder-time", 4000)   # 允许更大的乱序窗口(毫秒)
                src.set_property("max-dropout-time", 120000)  # 更大的掉包容忍时间(毫秒)
            except Exception as e:
                print("[tune] set RTPSource props failed:", e)
        session.connect("on-ssrc-validated", on_validated)

    # 2.2 每次创建 jitterbuffer 时降低启动门槛（见 faststart-min-packets）
    def on_new_jbuf(_rtpbin, jbuf, session_id, ssrc, *a):
        try:
            jbuf.set_property("faststart-min-packets", 1)     # 1个连续包就开始出队
            jbuf.set_property("max-misorder-time", 4000)
            jbuf.set_property("max-dropout-time", 120000)
        except Exception as e:
            print("[tune] set jbuf props failed:", e)
    try:
        rtpbin.connect("new-jitterbuffer", on_new_jbuf)
    except Exception:
        pass

def _rtp_h264_to_appsink_desc(appsink_name: str = "recv_app", to_format: str = "RGBA") -> str:
    """RTP H264 → RGBA → appsink (low-latency) with NV/CPU fallback.

    Keep a leaky queue before appsink to avoid backpressure.
    """
    # Prefer the NVDEC decoder when present; otherwise fall back to CPU avdec_h264.
    q = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
    dec = "nvh264dec" if Gst.ElementFactory.find("nvh264dec") else "avdec_h264"

    return (
        f'capsfilter caps="application/x-rtp" ! rtph264depay ! h264parse ! '
        f"{dec} ! {q} ! video/x-raw,format=NV12 ! videoconvert ! video/x-raw,format=RGB ! "
        f"appsink name={appsink_name} emit-signals=true sync=false max-buffers=1 drop=true"
    )


def _frame_from_sample(sample: Gst.Sample, expect_fmt: str = "RGBA") -> Tuple[np.ndarray, Tuple[int, int]]:
    """Map Gst.Sample → numpy array (H, W, C) with row-stride handling.

    Only minimal error handling; returns a copy to detach from Gst buffer.
    """
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0) if caps else None
    width = int(s.get_value("width")) if s and s.has_field("width") else 0
    height = int(s.get_value("height")) if s and s.has_field("height") else 0
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("buffer map failed")
    try:
        ch = 4 if expect_fmt.upper() in ("RGBA", "BGRA", "ARGB", "ABGR") else 3
        mv = memoryview(map_info.data)
        nbytes = len(mv)
        if height <= 0:
            raise RuntimeError("invalid frame dimensions")
        bpr = nbytes // height
        row_pixels = bpr // ch if ch > 0 else 0
        arr = np.frombuffer(mv, dtype=np.uint8)
        if ch == 4:
            arr = arr.reshape(height, row_pixels, 4)
            if width > 0 and row_pixels != width:
                arr = arr[:, :width, :]
        else:
            arr = arr.reshape(height, row_pixels, 3)
            if width > 0 and row_pixels != width:
                arr = arr[:, :width, :]
        return arr.copy(), (height, width)
    finally:
        buf.unmap(map_info)


class _InferWriter:
    """Own and protect a live InferenceBufferV2 instance.

    - Maintains per-stream pre-allocated tensors (uint8, CHW, RGB).
    - Writes with `tensor.copy_()` under a named semaphore.
    """

    def __init__(self, *, sem_name: Optional[str] = None) -> None:
        self.buf = InferenceBufferV2(images={}, meta={})
        # Create or open a named semaphore and expose the name for readers.
        self.sem_name = self.buf.attach_semaphore(name=sem_name, create=sem_name is None)
        print(f"[inferbuf] semaphore name: {self.sem_name}")
        # choose device dynamically to avoid hard dependency on CUDA
        # NOTE: CUDA tensors enable CUDA-IPC zero-copy; CPU fallback stays functional.
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def ensure_tensor(self, key: str, h: int, w: int) -> torch.Tensor:
        t = self.buf.images.get(key)
        if t is None or t.dtype != torch.uint8 or tuple(t.shape) != (h, w, 3):
            # Allocate uint8 CHW RGB on selected device
            self.buf.images[key] = torch.empty((h, w, 3), dtype=torch.uint8, device=self.device)
            t = self.buf.images[key]
        return t

    def write(self, key: str, frame: np.ndarray) -> None:
        # Keep HWC layout; drop alpha if present.
        h, w = frame.shape[:2]
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        dst = self.ensure_tensor(key, h, w)
        # from_numpy returns a CPU tensor; copy_ handles CPU→CUDA implicitly
        src = torch.from_numpy(frame)
        # Lock around a tight copy to keep critical section small.
        with self.buf.hold_lock(timeout=None):
            dst.copy_(src)  # in-place copy
            # Optional: update minimal meta for observability
            self.buf.meta["last_key"] = key


async def main(streams: int, host: str, port: int) -> None:
    load_env()
    init_gst()
    receiver = WebRTCReceiver(latency_ms=80)
    # relax_rtp_tolerance(receiver)
    receiver.set_rtp_sink_desc(_rtp_h264_to_appsink_desc())

    writer = _InferWriter()

    # Start a background Flask server to share the buffer (APIServer).
    provider = InferenceBufferProvider()
    provider.set_buffer(writer.buf)  # share live buffer instance
    server = APIServer(provider, host=host, port=port)
    server.start()
    print(f"[flask] serving on http://{host}:{port} (sem={writer.sem_name})")

    # Start receiver loop
    recv_task = asyncio.create_task(receiver.run())

    # Dynamically discover and bind msid-renamed appsinks; never exit early
    sinks: Dict[str, Gst.Element] = {}

    def _collect(container: Gst.Bin) -> None:
        # recursively walk pipeline to find appsinks by name
        it = container.iterate_recurse()
        try:
            while True:
                try:
                    res, elem = it.next()  # type: ignore[attr-defined]
                except Exception:
                    break
                if res == Gst.IteratorResult.OK:
                    if isinstance(elem, GstApp.AppSink):
                        nm = elem.get_name() or "appsink"
                        sinks.setdefault(nm, elem)
                    continue
                if res == Gst.IteratorResult.DONE:
                    break
                it = container.iterate_recurse()
        finally:
            it = None

    # Connect callbacks (idempotent per key)
    def on_new_sample(sink: Gst.Element, key: str):  # -> Gst.FlowReturn
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            # Expect RGB (3 channels) from appsink and keep HWC here
            arr, _ = _frame_from_sample(sample, expect_fmt="RGB")
            arr = resize_with_pad(arr, 224, 224)
            print(f"[{key}] frame {arr.shape} from appsink")
            writer.write(key, arr)
        except Exception:
            # keep pipeline alive on any conversion error
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.OK

    bound: Dict[str, bool] = {}

    print("[receiver] waiting for appsinks (msid) …")
    try:
        while True:
            _collect(receiver.pipe)
            # Bind any newly discovered sinks
            for key, sink in list(sinks.items()):
                if bound.get(key):
                    continue
                try:
                    sink.set_property("emit-signals", True)
                    sink.set_property("sync", False)
                    sink.set_property("max-buffers", 1)
                    sink.set_property("drop", True)
                except Exception:
                    pass
                sink.connect("new-sample", lambda s, k=key: on_new_sample(s, k))
                bound[key] = True
                print(f"[receiver] appsink bound: {key}")

            # Optional: one-time log when expected count is reached
            # if streams > 0 and len(bound) == streams:
            #     print(f"[receiver] all expected streams discovered: {list(bound.keys())}")
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.stop()
            print("[flask] server stopped")
        except Exception:
            pass
        # Best-effort semaphore cleanup
        try:
            writer.buf.close_semaphore()
            writer.buf.unlink_semaphore()
        except Exception:
            pass
        try:
            receiver.pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            recv_task.cancel()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stream receiver → InferenceBufferV2")
    p.add_argument("--streams", type=int, default=2, help="expected number of streams (msid)")
    p.add_argument("--host", type=str, default=os.environ.get("FLASK_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("FLASK_PORT", "5082")))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.streams, args.host, args.port))
