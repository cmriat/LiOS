"""gst-webrtc throughput receiver — C-side counting (no per-frame Python).

webrtcbin -> rtph264depay -> NVDEC -> fpsdisplaysink. Throughput is counted inside
GStreamer and reported via the 'fps-measurements' signal (periodic, ~2/s), so the
measurement is NOT capped by the Python GIL (the old per-frame appsink path
saturated ~700fps; the raw NVENC->NVDEC pipeline does ~2600fps).

Env: ROOM, SIGNAL_URL, STUN, TURN, DURATION, WARMUP
"""
import asyncio
import os
import sys
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # noqa: E402

from gst_webrtc import init_gst  # noqa: E402
from gst_webrtc.receiver import WebRTCReceiver  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import FpsCollector, fpsdisplay_sink  # noqa: E402

DURATION = float(os.environ.get("DURATION", 30))
WARMUP = float(os.environ.get("WARMUP", 4))
QUEUE = "queue max-size-buffers=4 max-size-time=0 max-size-bytes=0 leaky=downstream"


def _sink_desc() -> str:
    dec = "nvh264dec" if Gst.ElementFactory.find("nvh264dec") else "avdec_h264"
    print(f"[gst-receiver] decode='{dec}' (C-side fps counting)", flush=True)
    return (
        f'capsfilter caps="application/x-rtp" ! rtph264depay ! h264parse ! {QUEUE} ! '
        f"{dec} name=viddec ! {QUEUE} ! {fpsdisplay_sink('fpsmeter')}"
    )


async def main() -> None:
    init_gst()
    rx = WebRTCReceiver()
    rx.set_rtp_sink_desc(_sink_desc())
    coll = FpsCollector("gst-webrtc", warmup_s=WARMUP)
    state = {"end": None}

    def _on_meas(sink, fps, droprate, avgfps):
        coll.on_measurement(sink, fps, droprate, avgfps)
        if state["end"] is None and fps > 0:
            state["end"] = time.time() + WARMUP + DURATION
            print("[gst-receiver] frames flowing, counting…", flush=True)
        return True

    task = asyncio.create_task(rx.run())
    connected = False
    probed = False
    _dc = {"n": 0, "t0": None}

    def _probe(pad, info, _dc=_dc):
        if _dc["t0"] is None:
            _dc["t0"] = time.time()
        _dc["n"] += 1
        if _dc["n"] % 500 == 0:
            print(f"[gst-receiver] 真实进解码器 {_dc['n']}, {_dc['n']/(time.time()-_dc['t0']):.0f} pre-dec-fps", flush=True)
        return Gst.PadProbeReturn.OK

    print(f"[gst-receiver] warmup={WARMUP}s measure={DURATION}s", flush=True)
    while True:
        if not connected:
            sink = rx.pipe.get_by_name("fpsmeter")
            if sink is not None:
                sink.connect("fps-measurements", _on_meas)
                connected = True
                print("[gst-receiver] fpsdisplaysink connected", flush=True)
        # 帧开始流动后(活跃解码 bin 已建)再给所有 viddec 装探针, 数真实唯一帧
        if state["end"] and not probed and os.environ.get("DEPAY_COUNT"):
            probed = True
            it = rx.pipe.iterate_recurse()
            cnt = 0
            while True:
                ok, el = it.next()
                if ok != Gst.IteratorResult.OK:
                    break
                if el.get_name().startswith("viddec"):
                    sp = el.get_static_pad("sink")
                    if sp is not None:
                        sp.add_probe(Gst.PadProbeType.BUFFER, _probe)
                        cnt += 1
            print(f"[gst-receiver] 给 {cnt} 个 viddec 装了真实计数探针", flush=True)
        if state["end"] and time.time() > state["end"]:
            break
        await asyncio.sleep(0.1)

    coll.print_summary()
    try:
        rx.pipe.set_state(Gst.State.NULL)
    except Exception:
        pass
    task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
