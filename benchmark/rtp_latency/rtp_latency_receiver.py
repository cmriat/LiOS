#!/usr/bin/env python3
"""
RTP latency receiver: receives L16 audio RTP and computes one-way latency by
reading the first 8 bytes (little-endian ns) of each decoded buffer.

Usage (envs):
  LISTEN_PORT=5004 python benchmark/rtp_latency/rtp_latency_receiver.py

Note: This measures one-way latency and assumes clocks are comparable if
sender and receiver run on different hosts. For precise OWD, sync clocks.
"""
import os
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


LISTEN_PORT = int(os.environ.get("LISTEN_PORT", "5004"))
RATE = int(os.environ.get("RATE", "8000"))


Gst.init(None)


def build_pipeline():
    caps = (
        f"application/x-rtp,media=audio,clock-rate={RATE},encoding-name=L16,channels=1"
    )
    desc = f"""
udpsrc address=0.0.0.0 port={LISTEN_PORT} caps="{caps}" ! 
  rtpjitterbuffer mode=1 latency=50 drop-on-late=true do-retransmission=false ! 
  rtpL16depay ! 
  appsink name=asink emit-signals=true sync=false max-buffers=5 drop=true caps=audio/x-raw,format=S16LE,rate={RATE},channels=1
"""
    return Gst.parse_launch(desc)


def on_new_sample(appsink):

    sample = appsink.emit("pull-sample")
    if not sample:
        return Gst.FlowReturn.ERROR
    buf = sample.get_buffer()
    ok, mapinfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return Gst.FlowReturn.ERROR
    try:
        b = bytes(mapinfo.data)
    finally:
        buf.unmap(mapinfo)

    if len(b) < 8:
        return Gst.FlowReturn.OK

    sent_ns = int.from_bytes(b[:8], byteorder="little", signed=False)
    now_ns = time.monotonic_ns()
    owd_ms = (now_ns - sent_ns) / 1e6

    # Print a compact line; user can feed to awk/plot
    print(f"owd_ms={owd_ms:.3f}")
    return Gst.FlowReturn.OK


def main():
    pipe = build_pipeline()
    sink = pipe.get_by_name("asink")
    sink.connect("new-sample", on_new_sample)
    pipe.set_state(Gst.State.PLAYING)
    try:
        GLib.MainLoop().run()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()

