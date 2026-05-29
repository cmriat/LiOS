from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # type: ignore


@dataclass
class Frame:
    array: np.ndarray
    pts_ns: Optional[int]
    dts_ns: Optional[int]
    seqnum: Optional[int]
    width: int
    height: int
    format: str


def _plugin_available(name: str) -> bool:
    return bool(Gst.ElementFactory.find(name))


class GpuFrameSink:
    """
    Appsink-backed GPU→CPU frame sink with a thread-safe queue.

    Minimal implementation per docs/design/gpu-sink.md to get basic flow running.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        output_format: str = "RGBA",
        queue_size: int = 4,
        drop_when_full: bool = True,
    ) -> None:
        if output_format not in {"RGBA", "RGB", "BGR", "GRAY8"}:
            raise ValueError("output_format must be one of RGBA|RGB|BGR|GRAY8")
        self.name = name or f"appsink-{uuid.uuid4().hex[:8]}"
        self.output_format = output_format
        self.drop_when_full = drop_when_full

        self._q: "queue.Queue[Frame]" = queue.Queue(maxsize=max(1, queue_size))
        self._sink: Optional[Gst.Element] = None
        self._sig_id: Optional[int] = None

        # Stats
        self._dropped = 0
        self._errors = 0
        self._last_pts_ns: Optional[int] = None
        self._backend = self._detect_backend()
        self._lock = threading.Lock()

    # -------------------------- Public API --------------------------
    def rtp_h264_sink_desc(self) -> str:
        """
        Return a gst description string decoding RTP H264 to appsink.

        Chooses nvcodec when available, otherwise CPU fallback.
        """
        queue_cfg = (
            "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
        )
        tail = self.appsink_tail_desc()
        desc_nv = (
            f"capsfilter caps=\"application/x-rtp\" ! rtph264depay ! h264parse ! "
            f"{queue_cfg} ! nvh264dec ! {queue_cfg} ! {tail}"
        )
        desc_cpu = (
            f"capsfilter caps=\"application/x-rtp\" ! rtph264depay ! h264parse ! "
            f"{queue_cfg} ! avdec_h264 ! {queue_cfg} ! {tail}"
        )
        return desc_nv if self._is_nv_pipeline() else desc_cpu

    def appsink_tail_desc(self) -> str:
        """Return only the color/memory conversion tail + appsink."""
        # Prefer NVIDIA path when available; otherwise videoconvert fallback
        # Force a leaky queue right before appsink to ensure only latest frame is visible.
        # This protects appsink from backpressure and keeps latency low.
        tail_queue_cfg = (
            "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"
        )
        if self._is_nv_pipeline():
            return (
                f"nvvideoconvert ! video/x-raw,format={self.output_format} ! "
                f"{tail_queue_cfg} ! "
                f"appsink name={self.name} emit-signals=true sync=false max-buffers=1 drop=true"
            )
        else:
            return (
                f"videoconvert ! video/x-raw,format={self.output_format} ! "
                f"{tail_queue_cfg} ! "
                f"appsink name={self.name} emit-signals=true sync=false max-buffers=1 drop=true"
            )

    def bind(self, pipeline: Gst.Pipeline) -> None:
        """Find appsink by name in `pipeline`, set props and connect callback."""
        sink = pipeline.get_by_name(self.name)
        if sink is None:
            raise RuntimeError(
                f"appsink '{self.name}' not found in pipeline; ensure bin added first"
            )
        # Ensure properties (pipeline description already sets them, but be explicit)
        try:
            sink.set_property("emit-signals", True)
            sink.set_property("sync", False)
            sink.set_property("max-buffers", 1)
            sink.set_property("drop", True)
            sink.set_property("enable-last-sample", False)
        except Exception:
            pass

        # Connect callback
        if self._sig_id is None:
            self._sig_id = sink.connect("new-sample", self._on_new_sample)
        self._sink = sink

    def pull(self, timeout: Optional[float] = None) -> Optional[Frame]:
        # Always return the freshest frame by draining stale ones.
        # NOTE: This implements a leaky-latest policy on the consumer side.
        try:
            latest = self._q.get(timeout=timeout)
        except queue.Empty:
            return None

        drained = 0
        while True:
            try:
                # Non-blocking drain of any additional queued frames
                latest = self._q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            with self._lock:
                self._dropped += drained
        return latest

    def flush(self) -> int:
        dropped = 0
        while True:
            try:
                self._q.get_nowait()
                dropped += 1
            except queue.Empty:
                break
        return dropped

    def close(self) -> None:
        if self._sink is not None and self._sig_id is not None:
            try:
                self._sink.disconnect(self._sig_id)
            except Exception:
                pass
        self._sig_id = None
        self._sink = None
        self.flush()

    def stats(self) -> dict:
        with self._lock:
            return {
                "queued": self._q.qsize(),
                "dropped": self._dropped,
                "errors": self._errors,
                "last_pts_ns": self._last_pts_ns,
                "backend": self._backend,
                "name": self.name,
                "format": self.output_format,
            }

    # ------------------------- Internals -------------------------
    def _detect_backend(self) -> str:
        nv = self._is_nv_pipeline()
        return "nv: h264dec+nvvideoconvert" if nv else "cpu: avdec_h264+videoconvert"

    def _is_nv_pipeline(self) -> bool:
        return _plugin_available("nvh264dec") and _plugin_available("nvvideoconvert")

    def _on_new_sample(self, sink: Gst.Element):  # -> Gst.FlowReturn
        try:
            sample = sink.emit("pull-sample")
            if not sample:
                return Gst.FlowReturn.OK
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0) if caps else None
            width = int(s.get_value("width")) if s and s.has_field("width") else 0
            height = int(s.get_value("height")) if s and s.has_field("height") else 0
            fmt = s.get_value("format") if s and s.has_field("format") else self.output_format

            ok, map_info = buffer.map(Gst.MapFlags.READ)
            if not ok:
                return Gst.FlowReturn.OK
            try:
                arr = self._map_to_numpy(map_info, width, height, str(fmt))
                # Always copy: zero-copy is unsafe without keeping the buffer mapped.
                arr = arr.copy()
            finally:
                buffer.unmap(map_info)

            pts_ns = int(buffer.pts) if buffer.pts != Gst.CLOCK_TIME_NONE else None
            dts_ns = int(buffer.dts) if buffer.dts != Gst.CLOCK_TIME_NONE else None
            try:
                seqnum = int(buffer.get_seqnum())  # type: ignore[attr-defined]
            except Exception:
                seqnum = None

            frame = Frame(
                array=arr,
                pts_ns=pts_ns,
                dts_ns=dts_ns,
                seqnum=seqnum,
                width=width,
                height=height,
                format=str(fmt),
            )

            with self._lock:
                self._last_pts_ns = pts_ns

            # Enqueue with drop policy (drop oldest to keep freshest)
            try:
                self._q.put_nowait(frame)
            except queue.Full:
                if self.drop_when_full:
                    try:
                        _ = self._q.get_nowait()  # drop oldest
                    except queue.Empty:
                        pass
                    try:
                        self._q.put_nowait(frame)
                    except queue.Full:
                        with self._lock:
                            self._dropped += 1
                else:
                    # Block until space available
                    self._q.put(frame)

        except Exception:
            with self._lock:
                self._errors += 1
            # Keep pipeline running
            return Gst.FlowReturn.OK

        return Gst.FlowReturn.OK

    def _channels_for_format(self, fmt: str) -> int:
        fmt = fmt.upper()
        if fmt in ("RGBA", "BGRA", "ARGB", "ABGR"):
            return 4
        if fmt in ("RGB", "BGR"):
            return 3
        if fmt in ("GRAY8", "GRAY8_LE", "GRAY8_BE"):
            return 1
        # Default conservative
        return 3

    def _map_to_numpy(self, map_info, width: int, height: int, fmt: str) -> np.ndarray:
        ch = self._channels_for_format(fmt)
        if height <= 0:
            raise RuntimeError("invalid frame dimensions")
        # map_info.data is a Python buffer protocol object
        mv = memoryview(map_info.data)
        nbytes = len(mv)
        # Compute stride and row pixels; handle padded rows by slicing to width
        bytes_per_row = nbytes // height
        row_pixels = bytes_per_row // ch if ch > 0 else 0
        if row_pixels <= 0:
            raise RuntimeError("invalid row_pixels computed from buffer")
        arr = np.frombuffer(mv, dtype=np.uint8)
        if ch == 1:
            arr = arr.reshape(height, bytes_per_row)
            if width > 0 and bytes_per_row != width:
                arr = arr[:, :width]
            return arr
        else:
            arr = arr.reshape(height, row_pixels, ch)
            if width > 0 and row_pixels != width:
                arr = arr[:, :width, :]
            # Convert to requested output order if needed
            fmt_u = fmt.upper()
            if fmt_u == "BGR" and self.output_format.upper() == "RGB":
                arr = arr[:, :, ::-1]
            elif fmt_u == "RGB" and self.output_format.upper() == "BGR":
                arr = arr[:, :, ::-1]
            # For RGBA we keep as-is
            return arr
