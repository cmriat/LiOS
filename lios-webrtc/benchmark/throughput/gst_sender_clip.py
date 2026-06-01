"""gst 发送端: 读"公共片段"(raw I420) 经 appsrc 循环推 -> cudaupload -> nvh264enc(显式码率)
-> RTP -> webrtcbin。与 LiveKit 端编码同一段、同码率, 公平对比。

Env: ROOM SIGNAL_URL STUN TURN W H FPS BITRATE_KBPS CLIP PEER_ID
"""

import asyncio
import os
import threading
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # noqa: E402

from gst_webrtc import init_gst  # noqa: E402
from gst_webrtc.sender import WebRTCSender  # noqa: E402

ROOM = os.environ.get("ROOM", "bench")
SIGNAL_URL = os.environ.get("SIGNAL_URL", "ws://127.0.0.1:18080/ws")
STUN = os.environ.get("STUN", "stun://stun.l.google.com:19302")
TURN = os.environ.get("TURN", "turn://USERNAME:PASSWORD@TURN_HOST:3478?transport=udp")
W = int(os.environ.get("W", 640))
H = int(os.environ.get("H", 480))
FPS = int(os.environ.get("FPS", 60))
BR = int(os.environ.get("BITRATE_KBPS", 10000))  # nvh264enc bitrate 单位 kbps
CLIP = os.environ.get("CLIP", "/tmp/clip_i420.raw")
PEER_ID = os.environ.get("PEER_ID", f"clip-{os.getpid()}")
FSZ = W * H * 3 // 2  # I420 帧字节数
APPSRC = "clipsrc"
QUEUE = "queue max-size-buffers=4 max-size-time=0 max-size-bytes=0 leaky=downstream"

init_gst()


def _desc(name: str, appsrc_name: str) -> str:
    return (
        f"appsrc name={appsrc_name} is-live=true do-timestamp=false format=time ! "
        f"video/x-raw,format=I420,width={W},height={H},framerate={FPS}/1 ! "
        f"identity name={name} silent=true ! videoconvert ! video/x-raw,format=NV12 ! {QUEUE} ! "
        f"cudaupload ! video/x-raw(memory:CUDAMemory) ! "
        f"nvh264enc bitrate={BR} rc-mode=cbr ! h264parse config-interval=-1 ! video/x-h264,alignment=au ! "
        f"{QUEUE} ! rtph264pay aggregate-mode=zero-latency pt=96 ! "
        f'capsfilter caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000"'
    )


def _start_push(appsrc) -> threading.Thread:
    with open(CLIP, "rb") as f:
        data = f.read()
    nfr = len(data) // FSZ

    def _run():
        for _ in range(200):
            if appsrc.get_state(0)[1] == Gst.State.PLAYING:
                break
            time.sleep(0.05)
        period = 1.0 / FPS
        period_ns = int(1_000_000_000 / FPS)
        nxt = time.time()
        t0 = nxt
        i = 0
        while True:
            off = (i % nfr) * FSZ
            buf = Gst.Buffer.new_wrapped(data[off : off + FSZ])
            buf.pts = i * period_ns
            buf.duration = period_ns
            ret = appsrc.emit("push-buffer", buf)
            if ret in (Gst.FlowReturn.FLUSHING, Gst.FlowReturn.EOS):
                break
            i += 1
            if i % 500 == 0:
                print(f"[clip-sender] pushed {i}, {i / (time.time() - t0):.0f} push-fps (target {FPS})", flush=True)
            nxt += period
            dt = nxt - time.time()
            if dt > 0:
                time.sleep(dt)

    t = threading.Thread(target=_run, name="clip-push", daemon=True)
    t.start()
    return t


sender = WebRTCSender(ROOM, PEER_ID, SIGNAL_URL, STUN, TURN)
h = sender.add_video_source_desc(_desc("cam0", APPSRC))
appsrc = h.bin.get_by_name(APPSRC)
appsrc.set_property("caps", Gst.Caps.from_string(f"video/x-raw,format=I420,width={W},height={H},framerate={FPS}/1"))
appsrc.set_property("max-buffers", 4)
appsrc.set_property("leaky-type", 2)
appsrc.set_property("block", False)
_start_push(appsrc)
print(
    f"[clip-sender] {W}x{H}@{FPS} br={BR}kbps clip={CLIP} nfr={os.path.getsize(CLIP) // FSZ} -> {SIGNAL_URL}",
    flush=True,
)
asyncio.run(sender.run())
