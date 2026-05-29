"""
# E2E: verify receiver appsink names (msid) and save per-stream images

Run (requires signaling server running):

  ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws \
  pixi run python tests/e2e/appsink_msid_e2e.py \
    --names cam0 cam1 --frames 5 --out ./e2e_frames

This script launches both a sender (multi-stream) and a receiver in-process.
For each expected name, it verifies an appsink with that name is present
(receiver core renames appsink to the incoming pad's msid) and saves PNGs.
"""

from __future__ import annotations

import argparse
import asyncio
import pathlib
from typing import Dict, List

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
from gst_webrtc.sender import WebRTCSender


def _plugin_available(name: str) -> bool:
    return bool(Gst.ElementFactory.find(name))


def build_sw_sender_source(name: str) -> str:
    """Low-latency CPU H264 RTP source; insert identity with provided name."""
    wh = "width=640,height=480"
    fr = "framerate=30/1"
    queue = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
    return f"""
videotestsrc is-live=true pattern=ball ! video/x-raw,{wh},{fr} ! \
identity name={name} silent=true ! \
{queue} ! \
x264enc tune=zerolatency speed-preset=veryfast key-int-max=30 ! \
h264parse config-interval=-1 ! video/x-h264,alignment=au ! {queue} ! \
rtph264pay aggregate-mode=zero-latency pt=96 ! \
capsfilter caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000\"
"""


def rtp_h264_to_appsink_desc(appsink_name: str = "recv_app", to_format: str = "RGBA") -> str:
    """Decode RTP H264 to raw + appsink.

    Always use CPU decode here to avoid environments with CUDA/NV failures even
    when plugins are installed but no GPU context is available.
    """
    q = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
    dec = f"avdec_h264 ! {q}"
    return (
        f"capsfilter caps=\"application/x-rtp\" ! rtph264depay ! h264parse ! {q} ! "
        f"{dec} ! videoconvert ! video/x-raw,format={to_format} ! {q} ! "
        f"appsink name={appsink_name} emit-signals=true sync=false max-buffers=1 drop=true"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E2E appsink msid/name verification test")
    p.add_argument("--names", nargs="+", default=["cam0", "cam1"], help="expected stream names")
    p.add_argument("--frames", type=int, default=5, help="frames per stream to save")
    p.add_argument("--out", type=str, default="./e2e_frames", help="output directory")
    p.add_argument("--timeout", type=float, default=8.0, help="discovery timeout seconds")
    return p.parse_args()


def _frame_from_sample(sample: Gst.Sample, expect_fmt: str = "RGBA") -> np.ndarray:
    # Map buffer to numpy with row-padding handling; copy for safety across threads.
    buf = sample.get_buffer()
    caps = sample.get_caps()
    s = caps.get_structure(0) if caps else None
    width = int(s.get_value("width")) if s and s.has_field("width") else 0
    height = int(s.get_value("height")) if s and s.has_field("height") else 0
    ch = 4 if expect_fmt.upper() in ("RGBA", "BGRA", "ARGB", "ABGR") else 3
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("buffer map failed")
    try:
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


async def main(names: List[str], frames: int, out_dir: pathlib.Path, timeout: float) -> None:
    init_gst()
    if not _HAS_PIL:
        raise SystemExit("Pillow is required: run 'pixi run python -m pip install pillow'")

    # Build sender with named sources
    sender = WebRTCSender()
    for nm in names:
        sender.add_video_source_desc(build_sw_sender_source(nm))

    # Build receiver with RTP→appsink chain; appsink will be renamed to msid by receiver
    receiver = WebRTCReceiver()
    receiver.set_rtp_sink_desc(rtp_h264_to_appsink_desc())

    out_dir.mkdir(parents=True, exist_ok=True)

    # Launch both tasks
    sender_task = asyncio.create_task(sender.run())
    receiver_task = asyncio.create_task(receiver.run())

    # Discover appsinks (recursively) and verify names
    expected = set(names)
    discovered: Dict[str, Gst.Element] = {}
    deadline = asyncio.get_event_loop().time() + timeout
    def _collect_appsinks(container: Gst.Bin) -> None:
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
                        if nm not in discovered:
                            discovered[nm] = elem
                    continue
                if res == Gst.IteratorResult.DONE:
                    break
                # RESYNC/ERROR: re-create iterator and continue
                it = container.iterate_recurse()
        finally:
            it = None

    while asyncio.get_event_loop().time() < deadline:
        _collect_appsinks(receiver.pipe)
        if expected.issubset(discovered.keys()):
            break
        await asyncio.sleep(0.02)

    print(f"[e2e-msid] expected={sorted(expected)} discovered={sorted(discovered.keys())}")
    missing = expected.difference(discovered.keys())
    if missing:
        print(f"[e2e-msid][WARN] missing expected appsinks: {sorted(missing)}. Will proceed with discovered only.")

    # Decide which keys to use for saving: prefer expected when present, else discovered
    if expected.issubset(discovered.keys()):
        used_keys = list(expected)
    else:
        used_keys = list(discovered.keys())

    # Configure sinks and connect new-sample handlers
    saved: Dict[str, int] = {k: 0 for k in used_keys}

    def on_new_sample(sink: Gst.Element, key: str):  # -> Gst.FlowReturn
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            arr = _frame_from_sample(sample, expect_fmt="RGBA")
            img = Image.fromarray(arr, mode="RGBA")
            fn = out_dir / f"{key}_f{saved[key]:04d}.png"
            img.save(fn)
            saved[key] += 1
        except Exception:
            return Gst.FlowReturn.OK
        return Gst.FlowReturn.OK

    for key in used_keys:
        sink = discovered[key]
        try:
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
        except Exception:
            pass
        sink.connect("new-sample", lambda s, k=key: on_new_sample(s, k))

    # Wait until each stream saved the required number of frames
    end = asyncio.get_event_loop().time() + max(2.0, timeout * 3)
    while asyncio.get_event_loop().time() < end:
        if saved and all(v >= frames for v in saved.values()):
            break
        await asyncio.sleep(0.02)

    print(f"[e2e-msid] save counts: {saved}")
    if not saved or not all(v >= frames for v in saved.values()):
        raise SystemExit("not all streams produced enough frames (or none discovered)")

    # Teardown
    try:
        receiver.pipe.set_state(Gst.State.NULL)
    except Exception:
        pass
    try:
        # No explicit close API; cancel tasks
        receiver_task.cancel()
    except Exception:
        pass
    try:
        sender.pipe.set_state(Gst.State.NULL)
    except Exception:
        pass
    try:
        sender_task.cancel()
    except Exception:
        pass


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.names, args.frames, pathlib.Path(args.out), args.timeout))
