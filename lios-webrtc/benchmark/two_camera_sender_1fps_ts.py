import asyncio
import os
import time
from typing import Optional

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # type: ignore

from gst_webrtc import init_gst
from gst_webrtc.sender.core import WebRTCSender


ROOM = os.environ.get("ROOM", "demo")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.l.google.com:19302")
TURN = os.environ.get("TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp")


# NOTE: keep caps unquoted inside description strings.
WH = "width=640,height=480"
FR = "framerate=1/1"  # 1 FPS as requested
QUEUE = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"


# Encoder: auto-detect nvh264enc, fall back to x264enc on GPUs w/o NVENC (e.g. H20Z).
Gst.init(None)
_ENCODER = os.environ.get("ENCODER") or ("nvh264enc" if Gst.ElementFactory.find("nvh264enc") else "x264enc")


def _enc_chain() -> str:
    if _ENCODER == "nvh264enc":
        return (
            "cudaupload ! video/x-raw(memory:CUDAMemory) ! "
            "nvh264enc ! h264parse config-interval=-1 ! video/x-h264,alignment=au"
        )
    return (
        "x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=2 ! "
        "h264parse config-interval=-1 ! video/x-h264,alignment=au"
    )


def _build_videotest_src(name: str, vsrc_name: Optional[str] = None) -> str:
    """videotestsrc → NV12 → H264 (RTP) at 1 FPS. Encoder picked at import time."""
    vname = vsrc_name or f"vsrc_{name}"
    return f"""
videotestsrc name={vname} is-live=true pattern=ball ! video/x-raw,{WH},{FR} ! \
identity name={name} silent=true ! videoconvert ! video/x-raw,format=NV12 ! \
{QUEUE} ! {_enc_chain()} ! \
rtph264pay aggregate-mode=zero-latency pt=96 ! \
capsfilter caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000\"
"""


def _find_element_recurse(bin_: Gst.Bin, name: str) -> Optional[Gst.Element]:
    # minimal recursive lookup by name for elements inside parsed bins
    it = bin_.iterate_recurse()
    try:
        while True:
            try:
                res, elem = it.next()  # type: ignore[attr-defined]
            except Exception:
                break
            if res == Gst.IteratorResult.OK:
                try:
                    if elem.get_name() == name:
                        return elem
                except Exception:
                    pass
                continue
            if res == Gst.IteratorResult.DONE:
                break
            it = bin_.iterate_recurse()
    finally:
        it = None
    return None


def _attach_src_ts_probe(pipe: Gst.Pipeline, elem_name: str, label: str) -> None:
    # attach a BUFFER pad-probe on videotestsrc src pad and print UNIX time
    elem = _find_element_recurse(pipe, elem_name)
    if not elem:
        print(f"[sender][warn] element not found: {elem_name}")
        return
    pad = elem.get_static_pad("src")
    if not pad:
        print(f"[sender][warn] src pad missing: {elem_name}")
        return

    def _on_buf(_pad, _info):
        ts = time.time()
        print(f"[ts][sender][{label}] {ts}")
        return Gst.PadProbeReturn.OK

    pad.add_probe(Gst.PadProbeType.BUFFER, _on_buf)


async def main() -> None:
    init_gst()

    sender = WebRTCSender(ROOM, "camera", SIGNAL_URL, STUN, TURN)
    num_cams = int(os.environ.get("NUM_CAMS", 2))
    # one or two streams; videotestsrc keeps pipelines identical in structure
    sender.add_video_source_desc(_build_videotest_src("cam0", vsrc_name="vsrc_cam0"))
    if num_cams >= 2:
        sender.add_video_source_desc(_build_videotest_src("cam1", vsrc_name="vsrc_cam1"))

    # Use sender.pipe to attach probes on source pads
    _attach_src_ts_probe(sender.pipe, "vsrc_cam0", "cam0")
    if num_cams >= 2:
        _attach_src_ts_probe(sender.pipe, "vsrc_cam1", "cam1")

    duration = float(os.environ.get("DURATION", 30))
    try:
        await asyncio.wait_for(sender.run(), timeout=duration)
    except asyncio.TimeoutError:
        print(f"[sender] reached {duration}s, stopping")


if __name__ == "__main__":
    asyncio.run(main())
