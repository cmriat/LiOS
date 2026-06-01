"""LiveKit latency receiver — control for two_camera_receiver_inferbuf_1fps_ts.py.

Subscribes to the video track; for each *decoded* frame: convert to RGBA, copy
into a CUDA tensor, torch.cuda.synchronize(), then take the timestamp. Prints
`[ts][lk-receiver][cam0] <t_recv>` at the same logical point as the gst receiver's
InferenceBufferV2 write (frame landed in a GPU buffer).

Run in the optional `livekit` pixi env:
  pixi run -e livekit python benchmark/livekit_receiver_ts.py
Env: LK_URL, LK_KEY, LK_SECRET, LK_ROOM, DURATION
"""

import asyncio
import os
import time

import numpy as np
import torch
from livekit import api, rtc

URL = os.environ.get("LK_URL", "ws://127.0.0.1:7880")
KEY = os.environ.get("LK_KEY", "devkey")
SECRET = os.environ.get("LK_SECRET", "secret")
ROOM = os.environ.get("LK_ROOM", "lat")
DURATION = float(os.environ.get("DURATION", 60))

_DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    stop = asyncio.Event()
    state = {"deadline": None, "dst": None, "n": 0}

    async def _consume(track: rtc.Track) -> None:
        stream = rtc.VideoStream(track)
        loop = asyncio.get_event_loop()
        async for ev in stream:  # each event = one decoded frame
            rgba = ev.frame.convert(rtc.VideoBufferType.RGBA)
            h, w = rgba.height, rgba.width
            arr = np.frombuffer(rgba.data, dtype=np.uint8).reshape(h, w, 4)
            src = torch.from_numpy(arr.copy())
            dst = state["dst"]
            if dst is None or tuple(dst.shape) != (h, w, 4):
                dst = torch.empty((h, w, 4), dtype=torch.uint8, device=_DEV)
                state["dst"] = dst
            dst.copy_(src)  # CPU -> CUDA
            if _DEV.type == "cuda":
                torch.cuda.synchronize()  # ensure GPU write truly completed
            t_recv = time.time()
            print(f"[ts][lk-receiver][cam0] {t_recv}", flush=True)
            state["n"] += 1
            if state["n"] == 1:
                print("[lk-sub] ✅ first frame decoded & written to CUDA", flush=True)
            if state["deadline"] is None:
                state["deadline"] = loop.time() + DURATION
            elif loop.time() > state["deadline"]:
                print(f"[lk-sub] reached {DURATION}s, stopping", flush=True)
                break
        stop.set()

    @room.on("track_subscribed")
    def _on_sub(track, publication, participant):  # noqa: ANN001
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            print(f"[lk-sub] video track subscribed: {publication.name}", flush=True)
            asyncio.create_task(_consume(track))

    # Connect to LiveKit Cloud with its default routing (its own PoPs).
    await room.connect(URL, _token("lk-sub"))
    print(f"[lk-sub] connected room={ROOM} measure={DURATION}s device={_DEV}", flush=True)
    try:
        await asyncio.wait_for(stop.wait(), timeout=DURATION + 40)
    except asyncio.TimeoutError:
        print("[lk-sub] timeout waiting for frames", flush=True)
    await room.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
