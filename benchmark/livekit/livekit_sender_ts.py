"""LiveKit latency sender (H264 forced) — control for two_camera_sender_1fps_ts.py.

Publishes 1 fps RGBA frames over a LiveKit room with H264 forced. Prints
`[ts][lk-sender][cam0] <t_send>` right before `source.capture_frame(...)`, i.e. the
capture instant — the same logical point as gst's videotestsrc src pad.

Run in the optional `livekit` pixi env:
  pixi run -e livekit python benchmark/livekit_sender_ts.py
Env: LK_URL, LK_KEY, LK_SECRET, LK_ROOM, W, H, FPS, DURATION
"""
import asyncio
import os
import time

import numpy as np
from livekit import api, rtc

URL = os.environ.get("LK_URL", "ws://127.0.0.1:7880")
KEY = os.environ.get("LK_KEY", "devkey")
SECRET = os.environ.get("LK_SECRET", "secret")
ROOM = os.environ.get("LK_ROOM", "lat")
W = int(os.environ.get("W", 640))
H = int(os.environ.get("H", 480))
FPS = int(os.environ.get("FPS", 1))  # 1 fps -> unambiguous same-second pairing
DURATION = float(os.environ.get("DURATION", 70))


def _token(identity: str) -> str:
    return (
        api.AccessToken(KEY, SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(api.VideoGrants(room_join=True, room=ROOM))
        .to_jwt()
    )


async def main() -> None:
    room = rtc.Room()
    # Connect to LiveKit Cloud with its default routing (its own PoPs).
    await room.connect(URL, _token("lk-pub"))
    source = rtc.VideoSource(W, H)
    track = rtc.LocalVideoTrack.create_video_track("cam0", source)
    opts = rtc.TrackPublishOptions(
        source=rtc.TrackSource.SOURCE_CAMERA,
        video_codec=rtc.VideoCodec.H264,  # force H264 to match gst-webrtc
    )
    await room.local_participant.publish_track(track, opts)
    print(f"[lk-pub] publishing H264 {W}x{H}@{FPS}fps room={ROOM} -> {URL}", flush=True)

    buf = np.zeros((H, W, 4), dtype=np.uint8)
    buf[..., 3] = 255
    period = 1.0 / FPS
    next_t = time.time()
    t0 = next_t
    n = 0
    while time.time() - t0 < DURATION:
        buf[..., :3] = 0
        x = (n * 7) % max(1, W - 40)
        buf[H // 2 - 20 : H // 2 + 20, x : x + 40, :3] = 255
        frame = rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, buf.tobytes())
        t_send = time.time()
        source.capture_frame(frame)
        print(f"[ts][lk-sender][cam0] {t_send}", flush=True)
        n += 1
        next_t += period
        dt = next_t - time.time()
        if dt > 0:
            await asyncio.sleep(dt)
    print(f"[lk-pub] done, {n} frames", flush=True)
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
