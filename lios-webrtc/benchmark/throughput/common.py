"""Shared throughput benchmark utilities.

The benchmark measures **figures/sec** = sustained *decoded* frames per second
delivered to the application/consumer, end-to-end (capture -> encode -> network
/TURN -> server -> decode -> consumer). Both the gst-webrtc and the LiveKit
benchmarks count frames at the same logical point: a decoded frame handed to the
application. A warmup window is skipped so we measure steady state, not ramp-up.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FpsMeter:
    """Counts frames and computes steady-state FPS after a warmup window."""

    label: str
    warmup_s: float = 5.0

    _t0: Optional[float] = None
    _start_measure: Optional[float] = None
    _last: Optional[float] = None
    _count_total: int = 0
    _count_measure: int = 0
    _intervals_ms: List[float] = field(default_factory=list)
    _per_sec: Dict[int, int] = field(default_factory=dict)

    def tick(self) -> None:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        self._count_total += 1
        if now - self._t0 < self.warmup_s:
            return
        if self._start_measure is None:
            self._start_measure = now
            self._last = now
            return
        self._count_measure += 1
        if self._last is not None:
            self._intervals_ms.append((now - self._last) * 1000.0)
        self._last = now
        self._per_sec[int(now)] = self._per_sec.get(int(now), 0) + 1

    def _pct(self, p: float) -> float:
        iv = sorted(self._intervals_ms)
        if not iv:
            return 0.0
        i = min(len(iv) - 1, int(round(p / 100.0 * (len(iv) - 1))))
        return iv[i]

    def summary(self) -> dict:
        dur = self._last - self._start_measure if (self._last is not None and self._start_measure is not None) else 0.0
        fps = self._count_measure / dur if dur > 0 else 0.0
        secs = sorted(self._per_sec.values())
        return {
            "label": self.label,
            "fps": round(fps, 2),
            "frames_measured": self._count_measure,
            "measure_seconds": round(dur, 3),
            "interframe_ms_p50": round(self._pct(50), 2),
            "interframe_ms_p95": round(self._pct(95), 2),
            "fps_min_1s": secs[0] if secs else 0,
            "fps_max_1s": secs[-1] if secs else 0,
            "frames_total_incl_warmup": self._count_total,
        }

    def print_summary(self) -> dict:
        s = self.summary()
        # Machine-readable line for run_compare.sh to grep.
        print("RESULT_JSON " + json.dumps(s), flush=True)
        print(
            f"[{s['label']}] figures/sec={s['fps']}  frames={s['frames_measured']} "
            f"over {s['measure_seconds']}s  interframe p50={s['interframe_ms_p50']}ms "
            f"p95={s['interframe_ms_p95']}ms  (1s-window min/max {s['fps_min_1s']}/{s['fps_max_1s']})",
            flush=True,
        )
        return s


def fpsdisplay_sink(name: str = "fpsmeter", interval_ms: int = 500) -> str:
    """C-side throughput sink: counts buffers in GStreamer, reports fps periodically
    via the 'fps-measurements' signal. No per-frame Python callback -> not capped by
    the GIL (the Python appsink path saturates ~700fps; the raw pipeline does ~2600)."""
    return (
        f'fpsdisplaysink name={name} video-sink="fakesink sync=false" '
        f"text-overlay=false sync=false signal-fps-measurements=true "
        f"fps-update-interval={interval_ms}"
    )


@dataclass
class FpsCollector:
    """Collects fpsdisplaysink 'fps-measurements' (C-side), skipping a warmup window.

    Connect on_measurement to the fpsdisplaysink's 'fps-measurements' signal:
        sink.connect("fps-measurements", collector.on_measurement)
    """

    label: str
    warmup_s: float = 4.0
    interval_ms: int = 500

    _t0: Optional[float] = None
    _fps: List[float] = field(default_factory=list)  # per-interval current fps (steady)
    _avg: float = 0.0

    def on_measurement(self, _sink, fps: float, droprate: float, avgfps: float) -> bool:
        now = time.time()
        if self._t0 is None:
            self._t0 = now
        self._avg = avgfps
        if now - self._t0 >= self.warmup_s and fps > 0:
            self._fps.append(fps)
        return True

    def _pct(self, p: float) -> float:
        v = sorted(self._fps)
        if not v:
            return 0.0
        i = min(len(v) - 1, int(round(p / 100.0 * (len(v) - 1))))
        return v[i]

    def summary(self) -> dict:
        v = self._fps
        mean = (sum(v) / len(v)) if v else 0.0
        dur = len(v) * (self.interval_ms / 1000.0)
        return {
            "label": self.label,
            "fps": round(mean, 1),
            "fps_p50": round(self._pct(50), 1),
            "fps_min_1s": round(min(v), 1) if v else 0,
            "fps_max_1s": round(max(v), 1) if v else 0,
            "samples": len(v),
            "avg_fps_cumulative": round(self._avg, 1),
            "interframe_ms_p50": round(1000.0 / mean, 3) if mean > 0 else 0.0,
            "interframe_ms_p95": round(1000.0 / self._pct(5), 3) if self._pct(5) > 0 else 0.0,
            "frames_measured": int(mean * dur),
            "measure_seconds": round(dur, 1),
            "counter": "c-side-fpsdisplaysink",
        }

    def print_summary(self) -> dict:
        import json

        s = self.summary()
        print("RESULT_JSON " + json.dumps(s), flush=True)
        print(
            f"[{s['label']}] figures/sec(C端)={s['fps']}  p50={s['fps_p50']} "
            f"min/max={s['fps_min_1s']}/{s['fps_max_1s']}  samples={s['samples']}  "
            f"(cumulative avg {s['avg_fps_cumulative']})",
            flush=True,
        )
        return s
