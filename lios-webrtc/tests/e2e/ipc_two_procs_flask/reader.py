from __future__ import annotations

import argparse
import base64
import logging
import pickle
import time
import urllib.error
import urllib.request
from typing import Optional

import torch

from gst_webrtc.inference_buffer_v2 import InferenceBufferV2


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _http_get(url: str, *, timeout: float = 2.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() if e.fp else b""


def _p(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="Base URL to the writer's Flask server, e.g., http://127.0.0.1:5081")
    p.add_argument("--v1", type=float, default=1.2345)
    p.add_argument("--v2", type=float, default=2.3456)
    args = p.parse_args(argv)

    _setup_logging()
    log = logging.getLogger("e2e.flask.reader")

    # Health check loop until writer announces buffer
    health = f"{args.url.rstrip('/')}/api/v1/healthz"
    t0 = time.time()
    while True:
        code, body = _http_get(health, timeout=2.0)
        if code == 200:
            _p(f"HEALTHZ 200 {body.decode('utf-8', 'ignore')}")
            break
        _p(f"HEALTHZ {code} waiting...")
        if time.time() - t0 > 30.0:
            _p("FAIL healthz timeout")
            return 3
        time.sleep(0.2)

    # Fetch base64 payload once
    b64_url = f"{args.url.rstrip('/')}/api/v1/infer-buffer/base64"
    code, body = _http_get(b64_url, timeout=5.0)
    if code != 200:
        _p(f"FAIL GET_BASE64 status={code}")
        return 4
    payload_b64 = body.decode("ascii").strip()
    _p("B64_LEN", len(payload_b64))

    # Verify plain pickle/base64 round-trip without importing the class
    plain = pickle.loads(base64.b64decode(payload_b64))
    if not isinstance(plain, dict) or "images" not in plain:
        _p("FAIL plain decode structure")
        return 5
    _p(f"PLAIN_OK keys={sorted(plain.keys())}")

    # Rebuild structured helper for ergonomics and posix_ipc helpers
    buf = InferenceBufferV2.from_base64(payload_b64)
    buf.attach_semaphore(create=False)
    t = buf.images["rgb"]

    _p(f"READY role=reader device={t.device} cuda={int(t.is_cuda)} shape={tuple(t.shape)}")

    # Phase 1: Read mean after writer's write
    with buf.hold_lock():
        if t.is_cuda:
            torch.cuda.synchronize()
        m1 = float(t.mean().item())
    _p(f"READ_V1 {m1:.6f}")

    if t.is_cuda:
        if abs(m1 - float(args.v1)) > 1e-5:
            _p(f"FAIL reader v1 mismatch expected={args.v1} got={m1}")
            return 6

    # Phase 2: Reader writes v2
    with buf.hold_lock():
        t.fill_(float(args.v2))
        if t.is_cuda:
            torch.cuda.synchronize()
        m2 = float(t.mean().item())
    _p(f"WROTE_V2 {m2:.6f}")

    _p("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
