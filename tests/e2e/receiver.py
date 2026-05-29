import asyncio
import os

from gst_webrtc import init_gst
from gst_webrtc.receiver import WebRTCReceiver

ROOM = os.environ.get("ROOM", "demo")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.example.com")
TURN = os.environ.get(
    "TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp"
)

init_gst()

queue = "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"

def h264_decode_bin_sink() -> str:
    desc = f"""
capsfilter caps="application/x-rtp" ! rtph264depay ! h264parse !
{queue} !
avdec_h264 ! {queue} !
videoconvert ! fakesink sync=false
"""
    return desc

receiver = WebRTCReceiver(ROOM, "receiver", SIGNAL_URL, STUN, TURN)
receiver.set_rtp_sink_desc(h264_decode_bin_sink())

asyncio.run(receiver.run())
