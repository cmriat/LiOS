"""LiveKit throughput subscriber (optional `livekit` pixi env).

Subscribes to the published video track; rtc.VideoStream yields *decoded* frames.
Counts figures/sec at the same logical point as the gst-webrtc receiver: a decoded
frame handed to the application. (LiveKit decodes on CPU via libwebrtc — noted in
the README as a fairness caveat vs gst-webrtc's NVDEC.)

Run:  pixi run -e livekit python benchmark/throughput/livekit_subscriber.py
Env:  LK_URL, LK_KEY, LK_SECRET, LK_ROOM, DURATION, WARMUP
"""
import asyncio
import os
import sys

from livekit import api, rtc

sys.path.insert(0, os.path.dirname(__file__))
from common import FpsMeter  # noqa: E402

URL = os.environ.get("LK_URL", "ws://127.0.0.1:7880")
KEY = os.environ.get("LK_KEY", "devkey")
SECRET = os.environ.get("LK_SECRET", "secret")
ROOM = os.environ.get("LK_ROOM", "bench")
DURATION = float(os.environ.get("DURATION", 30))
WARMUP = float(os.environ.get("WARMUP", 5))


def _token(identity: str) -> str:
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=ROOM))
        .to_jwt()
    )


async def main() -> None:
    meter = FpsMeter("livekit", warmup_s=WARMUP)
    room = rtc.Room()
    stop = asyncio.Event()
    deadline = {"t": None}

    async def _consume(track: rtc.Track) -> None:
        stream = rtc.VideoStream(track)
        loop = asyncio.get_event_loop()
        async for _ev in stream:  # each event = one decoded frame
            meter.tick()
            if deadline["t"] is None:
                deadline["t"] = loop.time() + WARMUP + DURATION
            elif loop.time() > deadline["t"]:
                break
        stop.set()

    @room.on("track_subscribed")
    def _on_sub(track, publication, participant):  # noqa: ANN001
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            asyncio.create_task(_consume(track))

    await room.connect(URL, _token("sub"))  # auto_subscribe defaults to True
    print(f"[lk-sub] connected room={ROOM} warmup={WARMUP}s measure={DURATION}s", flush=True)

    try:
        await asyncio.wait_for(stop.wait(), timeout=WARMUP + DURATION + 30)
    except asyncio.TimeoutError:
        print("[lk-sub] timeout waiting for frames", flush=True)

    meter.print_summary()
    dump = os.environ.get("DUMP")
    if dump:
        with open(dump, "w") as f:
            f.write("\n".join(f"{x:.4f}" for x in meter._intervals_ms))
        print(f"[lk-sub] dumped {len(meter._intervals_ms)} 帧间隔(ms) -> {dump}", flush=True)
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
