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

So the default config runs end-to-end with no real camera (GPU still required for NVENC).
"""

import asyncio
import os

from gst_webrtc import init_gst, load_env

load_env()

ROOM = os.environ.get("ROOM", "demo")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.example.com")
TURN = os.environ.get("TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp")
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "test")
CAMERAS = os.environ.get("CAMERAS", "cam0,cam1")

from gst_webrtc.sender import WebRTCSender  # noqa: E402

init_gst()

WH = "width=640,height=480"
QUEUE = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
NVENC = (
    "nvh264enc rc-mode=cbr bitrate=8000 preset=p1 tune=ultra-low-latency zerolatency=true "
    "bframes=0 rc-lookahead=0 i-adapt=false b-adapt=false vbv-buffer-size=8000"
)
RTP_TAIL = (
    'h264parse config-interval=-1 ! video/x-h264,alignment=au ! '
    'rtph264pay aggregate-mode=zero-latency pt=96 mtu=1200 ! '
    'capsfilter caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"'
)


def test_source(name: str, fps: str = "30") -> str:
    """videotestsrc → NVENC RTP. `identity name=<name>` tags the source for msid."""
    return (
        f"videotestsrc is-live=true pattern=ball ! video/x-raw,{WH},framerate={fps}/1 ! "
        f"identity name={name} silent=true ! videoconvert ! video/x-raw,format=NV12 ! "
        f"{QUEUE} ! cudaupload ! video/x-raw(memory:CUDAMemory) ! {NVENC} ! {RTP_TAIL}"
    )


def v4l2_source(name: str, device: str, fps: str = "30") -> str:
    """v4l2 camera → NVENC RTP."""
    return (
        f"v4l2src device={device} name=v4l2_{name} io-mode=mmap ! "
        f"video/x-raw,{WH},framerate={fps}/1,format=YUY2 ! "
        f"identity name={name} silent=true ! videoconvert ! video/x-raw,format=NV12 ! "
        f"{QUEUE} ! cudaupload ! video/x-raw(memory:CUDAMemory) ! {NVENC} ! {RTP_TAIL}"
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
            descs.append(v4l2_source(name.strip(), device.strip() or "/dev/video0", fps.strip() or "30"))
        else:
            name, _, fps = item.partition("@")
            descs.append(test_source(name.strip(), fps.strip() or "30"))
    return descs


sender = WebRTCSender(ROOM, "camera", SIGNAL_URL, STUN, TURN, latency_ms=80)
print(f"[sender] source={VIDEO_SOURCE} cameras={CAMERAS} room={ROOM} signal={SIGNAL_URL}")
for desc in build_sources():
    sender.add_video_source_desc(desc)
asyncio.run(sender.run())
