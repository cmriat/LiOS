from __future__ import annotations

import argparse
import sys

import torch

from gst_webrtc.inference_buffer_v2 import InferenceBufferV2


def _p(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--payload", required=True, help="Base64 payload from writer")
    args = p.parse_args(argv)

    buf = InferenceBufferV2.from_base64(args.payload)
    buf.attach_semaphore(create=False)

    t = buf.images["rgb"]
    _p(f"READY role=reader device={t.device} cuda={int(t.is_cuda)} shape={tuple(t.shape)}")

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0].upper()
        if cmd == "WRITE":
            val = float(parts[1])
            with buf.hold_lock():
                t.fill_(val)
                if t.is_cuda:
                    torch.cuda.synchronize()
                mean = float(t.mean().item())
            _p(f"WROTE {mean:.6f}")
        elif cmd == "READMEAN":
            with buf.hold_lock():
                if t.is_cuda:
                    torch.cuda.synchronize()
                mean = float(t.mean().item())
            _p(f"MEAN {mean:.6f}")
        elif cmd == "QUIT":
            break
        else:
            _p(f"ERR unknown_cmd {cmd}")

    try:
        buf.close_semaphore()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
