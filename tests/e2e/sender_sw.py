"""
# Software-encoder sender for E2E

Use x264enc (CPU) to avoid GPU dependency during CI/E2E runs.

Run:
  ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws pixi run python e2e/sender_sw.py
"""

import asyncio
import pathlib
import sys

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # type: ignore  # noqa: F401

# Ensure project root on sys.path when running from e2e/
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gst_webrtc import init_gst
from gst_webrtc.sender import WebRTCSender


def create_sw_source() -> str:
    # Low-latency CPU path; outputs application/x-rtp (H264)
    wh = "width=640,height=480"
    fr = "framerate=30/1"
    queue = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
    return f"""
videotestsrc is-live=true pattern=ball ! video/x-raw,{wh},{fr} ! {queue} ! \
x264enc tune=zerolatency speed-preset=veryfast key-int-max=30 ! \
h264parse config-interval=-1 ! video/x-h264,alignment=au ! {queue} ! \
rtph264pay aggregate-mode=zero-latency pt=96 ! \
capsfilter caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"
"""


async def main() -> None:
    init_gst()
    sender = WebRTCSender()
    sender.add_video_source_desc(create_sw_source())
    await sender.run()


if __name__ == "__main__":
    asyncio.run(main())
