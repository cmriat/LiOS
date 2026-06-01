"""
Receive two WebRTC H264 streams and write into InferenceBufferV2.
Print UNIX timestamp right after GPU write completes while holding a POSIX lock.

Run (aligned with examples):
- Start signaling (see project docs).
- Start sender (1fps, videotestsrc):
    ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws \
    pixi run python benchmark/two_camera_sender_1fps_ts.py
- Start receiver (expect 2 streams by default):
    ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws \
    pixi run python benchmark/two_camera_receiver_inferbuf_1fps_ts.py --streams 2
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from typing import Dict, Optional, Tuple

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # type: ignore

import numpy as np
import torch

from gst_webrtc import init_gst
from gst_webrtc.receiver import WebRTCReceiver
from gst_webrtc.inference_buffer_v2 import InferenceBufferV2


def _rtp_h264_to_appsink_desc(appsink_name: str = "recv_app", to_format: str = "RGBA") -> str:
    # lean rtp→decode→convert→appsink; keep leaky queues to avoid backpressure
    q = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
    nv_ok = bool(Gst.ElementFactory.find("nvh264dec")) and bool(Gst.ElementFactory.find("nvvideoconvert"))
    if nv_ok:
        dec_and_csc = f"nvh264dec ! {q} ! nvvideoconvert ! video/x-raw,format={to_format}"
    else:
        dec_and_csc = f"avdec_h264 ! {q} ! videoconvert ! video/x-raw,format={to_format}"
    return (
        f'capsfilter caps="application/x-rtp" ! rtph264depay ! h264parse ! '
        f"{q} ! {dec_and_csc} ! {q} ! "
        f"appsink name={appsink_name} emit-signals=true sync=false max-buffers=1 drop=true"
    )


def _frame_from_sample(sample: Gst.Sample, expect_fmt: str = "RGBA") -> Tuple[np.ndarray, Tuple[int, int]]:
    # minimal mapping → ndarray with stride handling
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
    # own a live buffer + a posix semaphore; print ts after GPU copy completes
    def __init__(self, *, sem_name: Optional[str] = None) -> None:
        self.buf = InferenceBufferV2(images={}, meta={})
        self.sem_name = self.buf.attach_semaphore(name=sem_name, create=sem_name is None)
        print(f"[inferbuf] semaphore name: {self.sem_name}")
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def ensure_tensor(self, key: str, h: int, w: int) -> torch.Tensor:
        t = self.buf.images.get(key)
        if t is None or t.dtype != torch.uint8 or tuple(t.shape) != (h, w, 4):
            self.buf.images[key] = torch.empty((h, w, 4), dtype=torch.uint8, device=self.device)
            t = self.buf.images[key]
        return t

    def write(self, key: str, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        dst = self.ensure_tensor(key, h, w)
        src = torch.from_numpy(frame)
        # lock → copy (CPU→CUDA if available) → print ts → unlock
        with self.buf.hold_lock(timeout=None):
            dst.copy_(src)
            if self.device.type == "cuda":
                torch.cuda.synchronize()  # ensure the GPU write truly completed
            print(f"[ts][receiver][{key}] {time.time()}")
            self.buf.meta["last_key"] = key


async def main(streams: int) -> None:
    init_gst()
    receiver = WebRTCReceiver()
    receiver.set_rtp_sink_desc(_rtp_h264_to_appsink_desc())

    writer = _InferWriter()

    recv_task = asyncio.create_task(receiver.run())
    sinks: Dict[str, Gst.Element] = {}

    def _collect(container: Gst.Bin) -> None:
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

    arrivals: Dict[str, int] = {}

    def on_new_sample(sink: Gst.Element, key: str):  # -> Gst.FlowReturn
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            arrivals[key] = arrivals.get(key, 0) + 1
            if arrivals[key] == 1:
                print(f"[receiver] ✅ first frame ARRIVED at appsink: {key}", flush=True)
            arr, _ = _frame_from_sample(sample, expect_fmt="RGBA")
            writer.write(key, arr)
        except Exception as e:  # surface the real error instead of swallowing
            print(f"[receiver] ❌ on_new_sample error ({key}): {e!r}", flush=True)
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.OK

    bound: Dict[str, bool] = {}
    duration = float(os.environ.get("DURATION", 30))
    t_end = None  # set when first sink is bound; stop after `duration` seconds

    print(f"[receiver] waiting for appsinks (msid) … will run ~{duration}s after first stream")
    try:
        while True:
            _collect(receiver.pipe)
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
                if t_end is None:
                    t_end = time.time() + duration
                print(f"[receiver] appsink bound: {key}")
            if streams > 0 and len(bound) == streams:
                print(f"[receiver] all expected streams discovered: {list(bound.keys())}")
            if t_end is not None and time.time() > t_end:
                print(f"[receiver] reached {duration}s, stopping")
                break
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            receiver.pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            recv_task.cancel()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stream receiver → InferenceBufferV2 with ts")
    p.add_argument("--streams", type=int, default=2, help="expected number of streams (msid)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.streams))
