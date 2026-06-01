"""
Receiver msid-named sinks → save images.

Usage (ensure signaling + tests/e2e/sender.py are running):

  ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws \
  pixi run python tests/e2e/receiver_msid_save.py --frames 5 --streams 2 --out ./outputs/frames_msid
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    _HAS_PIL = False

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # type: ignore

from gst_webrtc import init_gst
from gst_webrtc.receiver import WebRTCReceiver


def _rtp_h264_to_appsink_desc(appsink_name: str = "recv_app", to_format: str = "RGBA") -> str:
    # CPU decode for stability across environments (no GPU context needed)
    q = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
    dec = f"avdec_h264 ! {q}"
    return (
        f"capsfilter caps=\"application/x-rtp\" ! rtph264depay ! h264parse ! "
        f"{dec} ! videoconvert ! video/x-raw,format={to_format} ! {q} ! "
        f"appsink name={appsink_name} emit-signals=true sync=false max-buffers=1 drop=true"
    )


def _sanitize(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name) or "unnamed"


def _frame_from_sample(sample: Gst.Sample, expect_fmt: str = "RGBA") -> np.ndarray:
    # Map buffer to numpy with row-padding handling
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
        arr = arr.reshape(height, row_pixels, ch)
        if width > 0 and row_pixels != width:
            arr = arr[:, :width, :]
        return arr.copy()
    finally:
        buf.unmap(map_info)


async def main(frames: int, out_dir: Path, streams: int, timeout: float = 8.0, warmup: float = 1.5) -> None:
    init_gst()
    receiver = WebRTCReceiver()
    receiver.set_rtp_sink_desc(_rtp_h264_to_appsink_desc())

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[receiver-msid] expecting {streams} stream(s); warmup {warmup}s then saving {frames} frame(s) each → {out_dir}")

    # Start receiver loop
    recv_task = asyncio.create_task(receiver.run())

    # Discover msid-renamed appsinks recursively
    sinks: Dict[str, Gst.Element] = {}
    deadline = asyncio.get_event_loop().time() + timeout
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

    while len(sinks) < streams and asyncio.get_event_loop().time() < deadline:
        _collect(receiver.pipe)
        if len(sinks) >= streams:
            break
        await asyncio.sleep(0.02)

    if not sinks:
        print("[receiver-msid] no appsinks discovered; exiting")
        try:
            receiver.pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        recv_task.cancel()
        return

    print(f"[receiver-msid] discovered sinks: {list(sinks.keys())}")

    # Bind per-sink callbacks. Skip a per-sink warmup window so we save
    # steady-state frames (not the initial keyframe-wait / sensor-warmup ones),
    # and cap saves at `frames` per sink so a lagging sink can't make a fast one
    # save unbounded frames.
    saved: Dict[str, int] = {k: 0 for k in sinks.keys()}
    first_seen: Dict[str, float] = {}

    def on_new_sample(sink: Gst.Element, key: str):  # -> Gst.FlowReturn
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            now = time.monotonic()
            first_seen.setdefault(key, now)
            if now - first_seen[key] < warmup:
                return Gst.FlowReturn.OK  # warming up: drop, don't save
            if saved[key] >= frames:
                return Gst.FlowReturn.OK  # already saved enough for this sink
            arr = _frame_from_sample(sample, expect_fmt="RGBA")
            if _HAS_PIL:
                img = Image.fromarray(arr, mode="RGBA")
                fn = out_dir / f"{_sanitize(key)}_f{saved[key]:04d}.png"
                img.save(fn)
            saved[key] += 1
        except Exception:
            # keep pipeline alive
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.OK

    # Configure and connect signals
    for key, sink in sinks.items():
        try:
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
        except Exception:
            pass
        sink.connect("new-sample", lambda s, k=key: on_new_sample(s, k))

    # Wait until each sink has saved the requested number of frames
    end = asyncio.get_event_loop().time() + warmup + max(2.0, timeout * 3)
    while asyncio.get_event_loop().time() < end:
        if all(v >= frames for v in saved.values()):
            break
        await asyncio.sleep(0.02)

    print(f"[receiver-msid] save counts: {saved}")

    # Teardown
    try:
        receiver.pipe.set_state(Gst.State.NULL)
    except Exception:
        pass
    try:
        recv_task.cancel()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Receiver msid sink save test")
    p.add_argument("--frames", type=int, default=5, help="frames per sink to save")
    p.add_argument("--streams", type=int, default=2, help="expected number of streams")
    p.add_argument("--out", type=str, default="outputs/frames_msid", help="output directory")
    p.add_argument("--timeout", type=float, default=8.0, help="discovery timeout seconds")
    p.add_argument("--warmup", type=float, default=1.5, help="seconds to skip per sink before saving")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.frames, Path(args.out), args.streams, args.timeout, args.warmup))
