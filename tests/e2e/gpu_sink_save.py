"""
# GPU sink test: pull numpy frames and save with PIL

Run:
  ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws \
  pixi run python tests/e2e/gpu_sink_save.py --frames 5 --out ./frames_gpu

Prereqs:
  - Start signaling server (see project docs) and a sender (`tests/e2e/sender_sw.py`).
  - Pillow is recommended: `pixi run python -m pip install pillow` if missing.
"""

import argparse
import asyncio
from pathlib import Path

import numpy as np

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    Image = None  # type: ignore
    _HAS_PIL = False

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstWebRTC", "1.0")
gi.require_version("GstSdp", "1.0")
from gi.repository import Gst  # type: ignore

from gst_webrtc import init_gst
from gst_webrtc.gpu_sink import GpuFrameSink
from gst_webrtc.receiver import WebRTCReceiver


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU sink numpy+PIL test")
    p.add_argument("--frames", type=int, default=5, help="number of frames to save")
    p.add_argument("--out", type=str, default="./frames_gpu", help="output directory")
    p.add_argument(
        "--format",
        type=str,
        default="RGBA",
        choices=["RGBA", "RGB", "BGR", "GRAY8"],
        help="appsink output format",
    )
    p.add_argument("--timeout", type=float, default=2.0, help="pull timeout seconds")
    return p.parse_args()


async def main(nframes: int, out_dir: Path, out_fmt: str, timeout: float) -> None:
    # Init GStreamer and receiver
    init_gst()
    receiver = WebRTCReceiver()
    sink = GpuFrameSink(output_format=out_fmt, queue_size=4)
    receiver.set_rtp_sink_desc(sink.rtp_h264_sink_desc())

    # Ensure output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gpu-sink-test] saving {nframes} frames to: {out_dir}")

    # Start receiver loop
    recv_task = asyncio.create_task(receiver.run())

    # Wait for appsink to be added by receiver, then bind
    for _ in range(200):  # ~2s max
        el = receiver.pipe.get_by_name(sink.name)
        if el is not None:
            sink.bind(receiver.pipe)
            print(f"[gpu-sink-test] bound appsink: {sink.name}")
            break
        await asyncio.sleep(0.01)

    saved = 0
    try:
        while saved < nframes:
            frame = sink.pull(timeout=timeout)
            if frame is None:
                continue

            arr: np.ndarray = frame.array
            info = {
                "shape": arr.shape,
                "dtype": str(arr.dtype),
                "min": int(arr.min()),
                "max": int(arr.max()),
                "pts_ns": frame.pts_ns,
                "fmt": frame.format,
            }
            print(f"[gpu-sink-test] frame {saved}: {info}")

            # Save with PIL when available; otherwise emit a one-time hint
            if _HAS_PIL:
                img = None
                if out_fmt == "GRAY8":
                    img = Image.fromarray(arr, mode="L")
                elif out_fmt in ("RGB", "BGR"):
                    img_rgb = arr[:, :, ::-1] if out_fmt == "BGR" else arr
                    img = Image.fromarray(img_rgb, mode="RGB")
                else:  # RGBA
                    img = Image.fromarray(arr, mode="RGBA")
                fn = out_dir / f"frame_{saved:04d}.png"
                img.save(fn)
            else:
                if saved == 0:
                    print(
                        "[gpu-sink-test] PIL not available; install with 'pip install pillow' to save images"
                    )
            saved += 1
    finally:
        # Teardown
        try:
            sink.close()
        except Exception:
            pass
        try:
            receiver.pipe.set_state(Gst.State.NULL)
        except Exception:
            pass
        try:
            recv_task.cancel()
        except Exception:
            pass


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.frames, Path(args.out), args.format, args.timeout))

