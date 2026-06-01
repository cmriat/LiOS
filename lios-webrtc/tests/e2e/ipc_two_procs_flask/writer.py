from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Optional

import torch

from gst_webrtc.inference_buffer_v2 import InferenceBufferV2
from gst_webrtc.services.flask_api import APIServer, InferenceBufferProvider


def _setup_logging(verbosity: int = 1) -> None:
    level = logging.DEBUG if verbosity > 1 else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _p(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("FLASK_TEST_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("FLASK_TEST_PORT", "5081")))
    p.add_argument("--v1", type=float, default=1.2345, help="Value writer will write first")
    p.add_argument("--v2", type=float, default=2.3456, help="Value reader should write later; writer will observe")
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args(argv)

    _setup_logging(args.verbose)
    log = logging.getLogger("e2e.flask.writer")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.zeros((800, 800), dtype=torch.float32, device=device)

    buf = InferenceBufferV2(images={"rgb": t}, meta={"role": "writer"})
    sem_name = buf.attach_semaphore()
    payload_b64 = buf.to_base64()

    provider = InferenceBufferProvider()
    provider.set_buffer(buf)
    server = APIServer(provider, host=args.host, port=args.port)
    server.start()

    # Announce readiness and essential info for the orchestrator
    _p(f"READY role=writer device={t.device} cuda={int(t.is_cuda)} shape={tuple(t.shape)}")
    _p(f"SERVER http://{args.host}:{args.port}")
    _p("B64_LEN", len(payload_b64))
    _p("SEM", sem_name)

    # Phase 1: Writer writes v1
    with buf.hold_lock():
        t.fill_(float(args.v1))
        if t.is_cuda:
            torch.cuda.synchronize()
        mean1 = float(t.mean().item())
        _p(f"WROTE_V1 {mean1:.6f}")
        log.info("writer wrote v1=%.6f; mean=%.6f", args.v1, mean1)

    # Phase 2: Observe reader's write of v2
    target = float(args.v2)
    deadline = time.time() + 30.0
    observed = False
    while time.time() < deadline:
        with buf.hold_lock():
            if t.is_cuda:
                torch.cuda.synchronize()
            m = float(t.mean().item())
        if abs(m - target) <= 1e-5:
            _p(f"OBSERVED_V2 {m:.6f}")
            log.info("writer observed reader's v2=%.6f", m)
            observed = True
            break
        time.sleep(0.1)

    # Shutdown server gracefully
    server.stop()
    _p("SERVER_STOPPED")

    # Cleanup semaphore (best-effort)
    try:
        buf.close_semaphore()
        buf.unlink_semaphore()
    except Exception as e:  # pragma: no cover - best effort
        log.warning("semaphore cleanup error: %s", e)

    if t.is_cuda and not observed:
        _p("FAIL writer did not observe v2")
        return 2

    _p("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

