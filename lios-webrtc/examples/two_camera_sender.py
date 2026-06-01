"""Multi-camera WebRTC sender.

Config comes from the environment (or a project-root `.env`, auto-loaded):

    ROOM          room name, must match the receiver        (default: demo)
    SIGNAL_URL    signaling WebSocket URL                   (default: ws://127.0.0.1:18080/ws)
    STUN          stun://host:port
    TURN          turn://user:pass@host:port?transport=udp
    VIDEO_SOURCE  "test" (videotestsrc, no camera needed) | "v4l2"  (default: test)
    CAMERAS       test:  comma-separated names              (default: cam0,cam1)
                  v4l2:  name=/dev/videoN@fps, comma-separated
                         e.g. mid=/dev/video0@30,left=/dev/video4@25
    WIDTH/HEIGHT  capture resolution                        (default: 640 / 480)
    FPS           default framerate when not given per-camera (default: 30)
    ENCODER       "auto" | "nv" (NVENC) | "sw" (x264, no GPU)        (default: auto)
                  auto = NVENC when nvh264enc+cudaupload exist, else x264 software.

The default config runs end-to-end with no real camera; with ENCODER=sw (or auto
on a box without NVENC) it needs no GPU at all.
"""

import asyncio
import os

from gst_webrtc import init_gst, load_env

load_env()

ROOM = os.environ.get("ROOM", "demo")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.l.google.com:19302")
TURN = os.environ.get("TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp")
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "test")
CAMERAS = os.environ.get("CAMERAS", "cam0,cam1")
WIDTH = os.environ.get("WIDTH", "640")
HEIGHT = os.environ.get("HEIGHT", "480")
FPS = os.environ.get("FPS", "30")
ENCODER = os.environ.get("ENCODER", "auto")  # auto | nv | sw

from gst_webrtc.sender import WebRTCSender  # noqa: E402

init_gst()
from gi.repository import Gst  # noqa: E402  # available after init_gst()

WH = f"width={WIDTH},height={HEIGHT}"
QUEUE = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
NVENC = (
    "nvh264enc rc-mode=cbr bitrate=8000 preset=p1 tune=ultra-low-latency zerolatency=true "
    "bframes=0 rc-lookahead=0 i-adapt=false b-adapt=false vbv-buffer-size=8000"
)
X264 = "x264enc tune=zerolatency speed-preset=veryfast key-int-max=30 bitrate=8000"
RTP_TAIL = (
    "h264parse config-interval=-1 ! video/x-h264,alignment=au ! "
    "rtph264pay aggregate-mode=zero-latency pt=96 mtu=1200 ! "
    'capsfilter caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"'
)


def _use_nvenc() -> bool:
    if ENCODER == "nv":
        return True
    if ENCODER == "sw":
        return False
    # auto: need both the H.264 encoder and the CUDA uploader.
    return bool(Gst.ElementFactory.find("nvh264enc")) and bool(Gst.ElementFactory.find("cudaupload"))


def _encode_chain() -> str:
    """convert → encode tail: GPU NVENC when available, else CPU x264."""
    if _use_nvenc():
        return (
            f"videoconvert ! video/x-raw,format=NV12 ! {QUEUE} ! cudaupload ! video/x-raw(memory:CUDAMemory) ! {NVENC}"
        )
    return f"videoconvert ! video/x-raw,format=I420 ! {QUEUE} ! {X264}"


def test_source(name: str, fps: str = "30") -> str:
    """videotestsrc → H.264 RTP. `identity name=<name>` tags the source for msid."""
    return (
        f"videotestsrc is-live=true pattern=ball ! video/x-raw,{WH},framerate={fps}/1 ! "
        f"identity name={name} silent=true ! {_encode_chain()} ! {RTP_TAIL}"
    )


def v4l2_source(name: str, device: str, fps: str = "30") -> str:
    """v4l2 camera → H.264 RTP."""
    return (
        f"v4l2src device={device} name=v4l2_{name} io-mode=mmap ! "
        f"video/x-raw,{WH},framerate={fps}/1,format=YUY2 ! "
        f"identity name={name} silent=true ! {_encode_chain()} ! {RTP_TAIL}"
    )


def build_sources() -> list[str]:
    descs = []
    for item in CAMERAS.split(","):
        item = item.strip()
        if not item:
            continue
        if VIDEO_SOURCE == "v4l2":
            name, _, rest = item.partition("=")
            device, _, fps = rest.partition("@")
            descs.append(v4l2_source(name.strip(), device.strip() or "/dev/video0", fps.strip() or FPS))
        else:
            name, _, fps = item.partition("@")
            descs.append(test_source(name.strip(), fps.strip() or FPS))
    return descs


sender = WebRTCSender(ROOM, "camera", SIGNAL_URL, STUN, TURN, latency_ms=80)
print(
    f"[sender] source={VIDEO_SOURCE} encoder={'nv' if _use_nvenc() else 'sw'} "
    f"cameras={CAMERAS} room={ROOM} signal={SIGNAL_URL}"
)
for desc in build_sources():
    sender.add_video_source_desc(desc)
asyncio.run(sender.run())
