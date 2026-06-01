from __future__ import annotations

import sys

import torch

from gst_webrtc.inference_buffer_v2 import InferenceBufferV2


def _p(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.zeros((8, 8), dtype=torch.float32, device=device)

    buf = InferenceBufferV2(images={"rgb": t}, meta={"role": "writer"})
    sem_name = buf.attach_semaphore()
    payload_b64 = buf.to_base64()

    _p(f"READY role=writer device={t.device} cuda={int(t.is_cuda)} shape={tuple(t.shape)}")
    _p("B64", payload_b64)
    _p("SEM", sem_name)

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
        buf.unlink_semaphore()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
