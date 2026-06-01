"""LiveKit throughput publisher (optional `livekit` pixi env).

Pushes RGBA frames at a target FPS into a LiveKit room via the rtc SDK, mirroring
the gst-webrtc sender (videotestsrc-like moving pattern, same W/H/FPS). The
LiveKit server (SFU) forwards the encoded track to subscribers.

Run:  pixi run -e livekit python benchmark/throughput/livekit_publisher.py
Env:  LK_URL, LK_KEY, LK_SECRET, LK_ROOM, W, H, FPS, DURATION
"""

import asyncio
import os
import time

import numpy as np
from livekit import api, rtc

URL = os.environ.get("LK_URL", "ws://127.0.0.1:7880")
KEY = os.environ.get("LK_KEY", "devkey")
SECRET = os.environ.get("LK_SECRET", "secret")
ROOM = os.environ.get("LK_ROOM", "bench")
W = int(os.environ.get("W", 640))
H = int(os.environ.get("H", 480))
FPS = int(os.environ.get("FPS", 60))
DURATION = float(os.environ.get("DURATION", 45))


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
    await room.connect(URL, _token("pub"))
    source = rtc.VideoSource(W, H)
    track = rtc.LocalVideoTrack.create_video_track("cam0", source)
    maxfps = float(os.environ.get("MAXFPS", "0"))  # 0 = SDK default (~30fps real-time)
    if maxfps > 0:
        opts = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            video_codec=rtc.VideoCodec.H264,
            video_encoding=rtc.VideoEncoding(
                max_framerate=maxfps,
                max_bitrate=int(os.environ.get("MAXBR", "80000000")),
            ),
        )
    else:
        opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA)
    await room.local_participant.publish_track(track, opts)
    print(f"[lk-pub] publishing {W}x{H}@{FPS} room={ROOM} -> {URL}", flush=True)

    clip = os.environ.get("CLIP")
    period = 1.0 / FPS
    next_t = time.time()
    t0 = next_t
    n = 0
    if clip:
        fsz = W * H * 3 // 2  # I420
        with open(clip, "rb") as cf:
            data = cf.read()
        nfr = len(data) // fsz
        print(f"[lk-pub] clip={clip} nfr={nfr} (I420, 同一段)", flush=True)
        while time.time() - t0 < DURATION:
            off = (n % nfr) * fsz
            frame = rtc.VideoFrame(W, H, rtc.VideoBufferType.I420, data[off : off + fsz])
            source.capture_frame(frame)
            n += 1
            next_t += period
            dt = next_t - time.time()
            if dt > 0:
                await asyncio.sleep(dt)
    else:
        buf = np.zeros((H, W, 4), dtype=np.uint8)
        buf[..., 3] = 255  # opaque alpha
        while time.time() - t0 < DURATION:
            # moving white box on black -> real changing content for the encoder
            buf[..., :3] = 0
            x = (n * 7) % max(1, W - 40)
            buf[H // 2 - 20 : H // 2 + 20, x : x + 40, :3] = 255
            frame = rtc.VideoFrame(W, H, rtc.VideoBufferType.RGBA, buf.tobytes())
            source.capture_frame(frame)
            n += 1
            next_t += period
            dt = next_t - time.time()
            if dt > 0:
                await asyncio.sleep(dt)
    print(f"[lk-pub] done, captured {n} frames in ~{DURATION}s", flush=True)
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
