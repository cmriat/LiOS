import base64
import pickle

import torch
from torch import multiprocessing as mp


def _child_proc(payload_b64: str, conn):
    try:
        # Rebuild plain payload without importing the V2 stub
        plain = pickle.loads(base64.b64decode(payload_b64))
        rgb = plain["images"]["rgb"]

        # Send basic meta back
        conn.send((tuple(rgb.shape), str(rgb.dtype), str(rgb.device), bool(rgb.is_cuda)))

        # Step 1: Wait for write instruction and perform it in child
        tag, v1 = conn.recv()
        assert tag == "write1"
        rgb.fill_(float(v1))
        if rgb.is_cuda:
            torch.cuda.synchronize()
        conn.send(("done1", float(rgb.mean().item())))

        # Step 2: Parent writes, child verifies visibility
        tag, v2 = conn.recv()
        assert tag == "parent_written2"
        if rgb.is_cuda:
            torch.cuda.synchronize()
        mean2 = float(rgb.mean().item())
        conn.send(("seen2", mean2, abs(mean2 - float(v2)) < 1e-5))

        conn.close()
    except Exception as e:
        try:
            conn.send(("child_error", repr(e)))
        except Exception:
            pass
        raise


def main() -> int:
    # Prefer spawn to avoid forking CUDA contexts
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"
    shape = (4, 5)
    t = torch.zeros(shape, device=device, dtype=torch.float32)

    # Import V2 only in the parent/main to demonstrate child independence
    from gst_webrtc.inference_buffer_v2 import InferenceBufferV2

    buf = InferenceBufferV2(
        images={"rgb": t},
        states={},
        prev_actions={},
        meta={"frame_id": 1, "note": "v2-self-contained"},
    )
    payload_b64 = buf.to_base64()

    parent_conn, child_conn = mp.Pipe(duplex=True)
    proc = mp.Process(target=_child_proc, args=(payload_b64, child_conn), daemon=True)
    proc.start()

    # Child readiness
    shape_c, dtype_c, device_c, is_cuda_c = parent_conn.recv()
    print(f"child ready: shape={shape_c} dtype={dtype_c} device={device_c} is_cuda={is_cuda_c}")

    # Step 1: Child writes v1; parent verifies visibility (only for CUDA)
    v1 = 1.2345
    parent_conn.send(("write1", v1))
    msg = parent_conn.recv()
    if msg[0] == "child_error":
        print(f"FAIL: child raised: {msg[1]}")
        proc.terminate()
        proc.join(5)
        return 6
    tag, mean1 = msg
    assert tag == "done1"
    if is_cuda:
        torch.cuda.synchronize()
        ok1 = torch.allclose(t, torch.full_like(t, v1))
        print(f"child->parent CUDA propagation: child_mean={mean1:.6f} parent_allclose={ok1}")
        if not ok1:
            print("FAIL: Parent did not observe child's CUDA write.")
            proc.terminate()
            proc.join(5)
            return 2
    else:
        print(f"child wrote mean={mean1:.6f} on CPU (sharing not required)")

    # Step 2: Parent writes v2; child should observe (only for CUDA)
    v2 = 2.3456
    t.fill_(v2)
    if is_cuda:
        torch.cuda.synchronize()
    parent_conn.send(("parent_written2", v2))
    msg = parent_conn.recv()
    if msg[0] == "child_error":
        print(f"FAIL: child raised: {msg[1]}")
        proc.terminate()
        proc.join(5)
        return 7
    tag, mean2, ok2 = msg
    assert tag == "seen2"
    if is_cuda:
        print(f"parent->child CUDA propagation: child_mean={mean2:.6f} child_allclose={ok2}")
        if not ok2:
            print("FAIL: Child did not observe parent's CUDA write.")
            proc.terminate()
            proc.join(5)
            return 3
    else:
        print(f"parent wrote v2={v2:.4f} on CPU; child_mean={mean2:.6f} (no share check)")

    parent_conn.close()
    proc.join(10)
    if proc.exitcode not in (0, None):
        print(f"Child process non-zero exit code: {proc.exitcode}")
        return 4

    print("PASS: InferenceBufferV2 base64 payload reconstructed without stub; CUDA IPC verified when available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
