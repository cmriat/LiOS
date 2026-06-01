"""gst-webrtc throughput sender: videotestsrc -> NVENC(H264) -> RTP -> webrtcbin.

Single stream, configurable resolution/framerate. Push target FPS as high as you
want to probe the sustainable decode throughput on the receiver side.

Env: ROOM, SIGNAL_URL, STUN, TURN, W, H, FPS, PEER_ID
"""

import asyncio
import os

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # noqa: E402  type: ignore

from gst_webrtc import init_gst  # noqa: E402
from gst_webrtc.sender import WebRTCSender  # noqa: E402

ROOM = os.environ.get("ROOM", "bench")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.l.google.com:19302")
TURN = os.environ.get("TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp")
W = int(os.environ.get("W", 640))
H = int(os.environ.get("H", 480))
FPS = int(os.environ.get("FPS", 60))
PEER_ID = os.environ.get("PEER_ID", f"gstbench-{os.getpid()}")

QUEUE = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"

# Encoder: auto-detect nvh264enc, fall back to x264enc on GPUs w/o NVENC (e.g. H20Z).
import gi as _gi  # noqa: E402

_gi.require_version("Gst", "1.0")
from gi.repository import Gst as _Gst  # noqa: E402

_Gst.init(None)
_ENCODER = os.environ.get("ENCODER") or ("nvh264enc" if _Gst.ElementFactory.find("nvh264enc") else "x264enc")


def _enc_chain() -> str:
    if _ENCODER == "nvh264enc":
        return (
            f"cudaupload ! video/x-raw(memory:CUDAMemory) ! "
            f"nvh264enc ! h264parse config-interval=-1 ! video/x-h264,alignment=au"
        )
    # software fallback: x264enc, low-latency tuning, baseline-ish at ~4 Mbps
    return (
        f"x264enc tune=zerolatency speed-preset=ultrafast bitrate=4000 key-int-max={FPS * 2} ! "
        f"h264parse config-interval=-1 ! video/x-h264,alignment=au"
    )


def _src(name: str) -> str:
    return (
        f"videotestsrc name=vsrc is-live=true pattern=ball ! "
        f"video/x-raw,width={W},height={H},framerate={FPS}/1 ! "
        f"identity name={name} silent=true ! videoconvert ! video/x-raw,format=NV12 ! "
        f"{QUEUE} ! {_enc_chain()} ! "
        f"rtph264pay aggregate-mode=zero-latency pt=96 ! "
        f'capsfilter caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"'
    )


async def main() -> None:
    init_gst()
    sender = WebRTCSender(ROOM, PEER_ID, SIGNAL_URL, STUN, TURN)
    sender.add_video_source_desc(_src("cam0"))
    print(f"[gst-sender] room={ROOM} {W}x{H}@{FPS} peer={PEER_ID} -> {SIGNAL_URL}", flush=True)
    await sender.run()


if __name__ == "__main__":
    asyncio.run(main())
