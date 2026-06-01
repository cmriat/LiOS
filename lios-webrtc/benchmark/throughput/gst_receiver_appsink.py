"""gst 接收端 (appsink 逐帧硬数, 可靠). 替代 fpsdisplaysink 计数(对成簇流虚高)。
连上管线里所有 appsink, 各自计数 + 总数, 兼作多解码链诊断。
~700fps 以下计数准(GIL); 片段测试帧率远低于此。
Env: ROOM SIGNAL_URL STUN TURN DURATION WARMUP
"""

import asyncio
import os
import time

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst, GstApp  # noqa: E402

from gst_webrtc import init_gst  # noqa: E402
from gst_webrtc.receiver import WebRTCReceiver  # noqa: E402

DURATION = float(os.environ.get("DURATION", 8))
WARMUP = float(os.environ.get("WARMUP", 3))
QUEUE = "queue max-size-buffers=8 max-size-time=0 max-size-bytes=0 leaky=no"


def _sink_desc() -> str:
    dec = "nvh264dec" if Gst.ElementFactory.find("nvh264dec") else "avdec_h264"
    print(f"[recv-appsink] decode='{dec}' (appsink 逐帧硬数)", flush=True)
    return (
        f'capsfilter caps="application/x-rtp" ! rtph264depay ! h264parse ! {QUEUE} ! '
        f"{dec} ! {QUEUE} ! appsink name=cnt emit-signals=true sync=false max-buffers=8 drop=false"
    )


async def main() -> None:
    init_gst()
    rx = WebRTCReceiver()
    rx.set_rtp_sink_desc(_sink_desc())
    st = {"t_first": None, "t_ms": None, "measured": 0, "end": None, "last": None, "ivals": []}
    per = {}

    def mk_cb(name):
        def _cb(sink):
            s = sink.emit("pull-sample")
            if s is None:
                return Gst.FlowReturn.OK
            now = time.time()
            if st["t_first"] is None:
                st["t_first"] = now
                print("[recv-appsink] 首帧到达", flush=True)
            per[name] = per.get(name, 0) + 1
            if now - st["t_first"] >= WARMUP:
                if st["t_ms"] is None:
                    st["t_ms"] = now
                if st["last"] is not None:
                    st["ivals"].append((now - st["last"]) * 1000.0)
                st["last"] = now
                st["measured"] += 1
                if now - st["t_ms"] >= DURATION and st["end"] is None:
                    st["end"] = now
            return Gst.FlowReturn.OK

        return _cb

    task = asyncio.create_task(rx.run())
    seen = set()
    deadline = time.time() + WARMUP + DURATION + 40
    print(f"[recv-appsink] warmup={WARMUP}s measure={DURATION}s", flush=True)
    while time.time() < deadline:
        it = rx.pipe.iterate_recurse()
        while True:
            ok, el = it.next()
            if ok != Gst.IteratorResult.OK:
                break
            if isinstance(el, GstApp.AppSink):
                nm = el.get_name()
                if nm not in seen:
                    seen.add(nm)
                    el.set_property("emit-signals", True)
                    el.connect("new-sample", mk_cb(nm))
                    print(f"[recv-appsink] 连上 appsink: {nm} (共 {len(seen)} 个)", flush=True)
        if st["end"] is not None:
            break
        await asyncio.sleep(0.05)

    secs = (st["end"] - st["t_ms"]) if (st["t_ms"] and st["end"]) else DURATION
    fps = st["measured"] / secs if secs > 0 else 0
    print(f"[recv-appsink] appsink 个数={len(seen)} 各自计数={per}", flush=True)
    print(
        f'RESULT_JSON {{"label":"gst-appsink","fps":{fps:.1f},'
        f'"frames_measured":{st["measured"]},"measure_seconds":{secs:.2f},"appsinks":{len(seen)}}}',
        flush=True,
    )
    dump = os.environ.get("DUMP")
    if dump:
        with open(dump, "w") as f:
            f.write("\n".join(f"{x:.4f}" for x in st["ivals"]))
        print(f"[recv-appsink] dumped {len(st['ivals'])} 帧间隔(ms) -> {dump}", flush=True)
    try:
        rx.pipe.set_state(Gst.State.NULL)
    except Exception:
        pass
    task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
