"""Unit tests for InferenceBufferV2 base64 transport and POSIX semaphores."""

import base64
import pickle

import pytest

torch = pytest.importorskip("torch")

from gst_webrtc.inference_buffer_v2 import (  # noqa: E402
    IFBUF_PKL_VERSION,
    InferenceBufferV2,
    pack_base64,
    unpack_base64,
)


def test_pack_unpack_roundtrip_cpu():
    img = torch.ones(2, 3)
    s = pack_base64(images={"cam0": img}, meta={"frame_id": 7})
    assert isinstance(s, str)
    out = unpack_base64(s)
    assert {"header", "meta", "images", "states", "prev_actions"} <= set(out)
    assert out["header"]["__ifbuf__"] == IFBUF_PKL_VERSION
    assert out["meta"]["frame_id"] == 7
    assert torch.equal(out["images"]["cam0"], img)


def test_header_has_created_ns():
    out = unpack_base64(pack_base64(images={"x": torch.zeros(1)}))
    assert isinstance(out["header"]["created_ns"], int)
    assert out["header"]["created_ns"] > 0


def test_meta_keys_coerced_to_str():
    payload = InferenceBufferV2(images={}, meta={1: "a"}).to_plain_payload()
    assert "1" in payload["meta"]


def test_to_plain_payload_keys():
    payload = InferenceBufferV2(images={"c": torch.zeros(1)}).to_plain_payload()
    assert set(payload) == {
        "header",
        "meta",
        "images",
        "states",
        "prev_actions",
        "sem_name",
        "sem_state_name",
    }


def test_from_base64_roundtrip():
    s = InferenceBufferV2(images={"cam": torch.ones(2, 2)}, meta={"m": 1}).to_base64()
    back = InferenceBufferV2.from_base64(s)
    assert isinstance(back, InferenceBufferV2)
    assert torch.equal(back.images["cam"], torch.ones(2, 2))
    assert back.meta["m"] == 1


def test_self_contained_unpickle_without_class():
    # The docstring promises a receiver can recover a plain dict with only
    # stdlib pickle + torch (no need to import this module).
    raw = base64.b64decode(pack_base64(images={"x": torch.zeros(3)}))
    obj = pickle.loads(raw)
    assert isinstance(obj, dict)
    assert "images" in obj


def test_metadata_alias_is_meta():
    buf = InferenceBufferV2(images={}, meta={"k": "v"})
    assert buf.metadata is buf.meta


def test_pack_base64_images_is_required():
    with pytest.raises(TypeError):
        pack_base64()


# --- POSIX named-semaphore behaviour (Linux only) ---

posix_ipc = pytest.importorskip("posix_ipc")


def test_semaphore_lock_unlock_cycle():
    buf = InferenceBufferV2(images={})
    name = buf.attach_semaphore()
    try:
        assert name.startswith("/")
        assert buf.try_lock() is True
        assert buf.is_locked is True
        buf.unlock()
        assert buf.is_locked is False
    finally:
        buf.close_semaphore()
        buf.unlink_semaphore()


def test_hold_lock_context_manager():
    buf = InferenceBufferV2(images={})
    buf.attach_semaphore()
    try:
        with buf.hold_lock():
            assert buf.is_locked is True
        assert buf.is_locked is False
    finally:
        buf.close_semaphore()
        buf.unlink_semaphore()
