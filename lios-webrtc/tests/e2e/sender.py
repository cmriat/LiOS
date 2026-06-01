import asyncio
import os
import threading
import time

import gi

gi.require_version("Gst", "1.0")

from gi.repository import Gst

from gst_webrtc import init_gst
from gst_webrtc.sender import WebRTCSender

ROOM = os.environ.get("ROOM", "demo")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.example.com")
TURN = os.environ.get(
    "TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp"
)

init_gst()

queue = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"


# RealSense D405 color stream parameters (pushed into GStreamer via appsrc).
RS_W, RS_H, RS_FPS = (
    int(os.environ.get("RS_W", 640)),
    int(os.environ.get("RS_H", 480)),
    int(os.environ.get("RS_FPS", 30)),
)
# Camera is physically mounted sideways; rotate to upright. method nicks:
# none / clockwise / counterclockwise / rotate-180 / horizontal-flip / vertical-flip
RS_FLIP = os.environ.get("FLIP", "clockwise")


def create_realsense_appsrc_desc(name: str, appsrc_name: str) -> str:
    """RealSense color frames (RGB8) are pushed via pyrealsense2 into an appsrc.

    appsrc(RGB) → videoconvert → nvh264enc → RTP. `identity name=<name>` carries
    the msid label, identical to the test source.
    """
    src = f"""\
appsrc name={appsrc_name} is-live=true do-timestamp=true format=time ! \
video/x-raw,format=RGB,width={RS_W},height={RS_H},framerate={RS_FPS}/1 ! \
videoconvert ! videoflip method={RS_FLIP} ! \
identity name={name} silent=true ! \
{queue} ! \
nvh264enc ! \
h264parse config-interval=-1 ! video/x-h264,alignment=au ! \
{queue} ! \
rtph264pay aggregate-mode=zero-latency pt=96 ! \
capsfilter caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000\"
"""
    return src


def start_realsense_capture(appsrc, width=RS_W, height=RS_H, fps=RS_FPS):
    """Background thread: pyrealsense2 color frames → appsrc.push-buffer."""
    import numpy as np
    import pyrealsense2 as rs

    def _run():
        rspipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        rspipe.start(cfg)
        print(f"[realsense] color started {width}x{height}@{fps} rgb8")
        # Wait until appsrc is PLAYING so do-timestamp uses valid running-time.
        for _ in range(200):  # ~10s max
            if appsrc.get_state(0)[1] == Gst.State.PLAYING:
                break
            time.sleep(0.05)
        pushed = 0
        try:
            while True:
                frames = rspipe.wait_for_frames()
                color = frames.get_color_frame()
                if not color:
                    continue
                data = np.asanyarray(color.get_data()).tobytes()
                buf = Gst.Buffer.new_wrapped(data)
                ret = appsrc.emit("push-buffer", buf)
                pushed += 1
                if pushed == 1 or pushed % 60 == 0:
                    print(f"[realsense] pushed {pushed} frames (last ret={ret.value_nick})")
                if ret in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.EOS):
                    print(f"[realsense] stop pushing, ret={ret.value_nick}")
                    break
        finally:
            try:
                rspipe.stop()
            except Exception:
                pass
            print("[realsense] capture stopped")

    t = threading.Thread(target=_run, name="realsense-capture", daemon=True)
    t.start()
    return t


RS_APPSRC = "rscam0"
# Unique peer id so two senders / restarts don't collide ("register failed").
PEER_ID = os.environ.get("PEER_ID", f"camera-{os.getpid()}")
sender = WebRTCSender(ROOM, PEER_ID, SIGNAL_URL, STUN, TURN)
# Single real source: cam0 = RealSense D405 color (via appsrc), upright.
h0 = sender.add_video_source_desc(create_realsense_appsrc_desc("cam0", RS_APPSRC))

# Tune the appsrc for a live source and wire the camera capture thread to it.
appsrc0 = h0.bin.get_by_name(RS_APPSRC)
appsrc0.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=RGB,width={RS_W},height={RS_H},framerate={RS_FPS}/1"))
appsrc0.set_property("max-buffers", 4)
appsrc0.set_property("leaky-type", 2)  # downstream: drop oldest when full
appsrc0.set_property("block", False)
start_realsense_capture(appsrc0)

asyncio.run(sender.run())
