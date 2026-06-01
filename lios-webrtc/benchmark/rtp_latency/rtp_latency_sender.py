#!/usr/bin/env python3
"""
RTP latency sender: sends small L16 audio packets where the first 8 bytes of
each audio buffer carry the sender's monotonic time in nanoseconds.

Usage (envs):
  DEST_HOST=127.0.0.1 DEST_PORT=5004 python benchmark/rtp_latency/rtp_latency_sender.py
"""

import asyncio
import os
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib


DEST_HOST = os.environ.get("DEST_HOST", "127.0.0.1")
DEST_PORT = int(os.environ.get("DEST_PORT", "5004"))
RATE = int(os.environ.get("RATE", "8000"))  # Hz
PTIME_MS = int(os.environ.get("PTIME_MS", "20"))  # per RTP packet duration


Gst.init(None)


PIPELINE = f"""
appsrc name=asrc is-live=true format=time do-timestamp=true caps=audio/x-raw,format=S16LE,rate={RATE},channels=1 ! \
  queue leaky=downstream max-size-buffers=4 ! audioconvert ! \
  rtpL16pay pt=97 ptime={PTIME_MS} ! udpsink host={DEST_HOST} port={DEST_PORT}
"""


async def main():
    pipe = Gst.parse_launch(PIPELINE)
    asrc = pipe.get_by_name("asrc")
    if not asrc:
        raise RuntimeError("appsrc not found")

    samples_per_packet = RATE * PTIME_MS // 1000
    bytes_per_packet = samples_per_packet * 2  # S16LE mono

    pipe.set_state(Gst.State.PLAYING)

    try:
        while True:
            buf = Gst.Buffer.new_allocate(None, bytes_per_packet, None)
            now_ns = time.monotonic_ns()
            # Write 8 bytes (little-endian) at the start
            data = bytearray(bytes_per_packet)
            data[:8] = now_ns.to_bytes(8, byteorder="little", signed=False)
            ok, mapinfo = buf.map(Gst.MapFlags.WRITE)
            if not ok:
                raise RuntimeError("Failed to map buffer for write")
            try:
                mapinfo.data[:bytes_per_packet] = data
            finally:
                buf.unmap(mapinfo)
            # Set timing metadata for completeness
            buf.pts = Gst.util_uint64_scale_round(now_ns, Gst.SECOND, 1_000_000_000)
            buf.dts = buf.pts
            buf.duration = Gst.util_uint64_scale_int(PTIME_MS, Gst.SECOND, 1000)

            flow = asrc.emit("push-buffer", buf)
            if flow != Gst.FlowReturn.OK:
                print("push-buffer flow:", flow)
            await asyncio.sleep(PTIME_MS / 1000)
    finally:
        asrc.emit("end-of-stream")
        pipe.set_state(Gst.State.NULL)


if __name__ == "__main__":
    asyncio.run(main())
