"""Pure-logic unit tests for WebRTCReceiver signaling decisions.

No GPU / no network / no signaling server: a bare instance is created via
__new__ and the GStreamer/signaling collaborators are replaced with fakes.
Skips where `gi` is unavailable (e.g. CPU-only CI without GStreamer).
"""

import json

import pytest

pytest.importorskip("gi")

from _fakes import FakeSignal, FakeWebRTC  # noqa: E402

from gst_webrtc.receiver.core import WebRTCReceiver  # noqa: E402
from gst_webrtc.ws_signal.signal_client import Envelope  # noqa: E402


def make_receiver() -> WebRTCReceiver:
    """Bare receiver with fake collaborators (skips the real pipeline build)."""
    r = object.__new__(WebRTCReceiver)
    r.signal = FakeSignal()
    r.webrtc = FakeWebRTC()
    r.remote_id = None
    r._remote_sdp_set = False
    r._pending_local_ice = []
    r._pending_remote_ice = []
    return r


# --------------------------- remote candidate ---------------------------

def test_remote_candidate_buffered_before_remote_sdp():
    r = make_receiver()
    r._remote_sdp_set = False
    env = Envelope(type="candidate", data={"candidate": "cand:1", "sdpMLineIndex": 2})
    r._handle_remote_candidate(env)
    assert r._pending_remote_ice == [(2, "cand:1")]
    assert r.webrtc.emitted == []  # nothing applied yet


def test_remote_candidate_applied_after_remote_sdp():
    r = make_receiver()
    r._remote_sdp_set = True
    env = Envelope(type="candidate", data={"candidate": "cand:1", "sdpMLineIndex": 2})
    r._handle_remote_candidate(env)
    assert r._pending_remote_ice == []
    assert r.webrtc.emitted == [("add-ice-candidate", (2, "cand:1"))]


def test_remote_candidate_accepts_json_string_payload():
    r = make_receiver()
    r._remote_sdp_set = True
    env = Envelope(type="candidate", data=json.dumps({"candidate": "c", "sdpMLineIndex": 0}))
    r._handle_remote_candidate(env)
    assert r.webrtc.emitted == [("add-ice-candidate", (0, "c"))]


def test_remote_candidate_none_is_ignored():
    r = make_receiver()
    r._remote_sdp_set = True
    r._handle_remote_candidate(Envelope(type="candidate", data={"candidate": None}))
    assert r.webrtc.emitted == []
    assert r._pending_remote_ice == []


# --------------------------- offer handling -----------------------------

def test_handle_offer_missing_sdp_is_ignored(monkeypatch):
    r = make_receiver()
    calls = []
    monkeypatch.setattr(r, "_set_remote_sdp", lambda *a, **k: calls.append("set"))
    monkeypatch.setattr(r, "_create_and_send_answer", lambda: calls.append("answer"))
    r._handle_offer(Envelope(type="offer", from_="peerA", data={}))
    assert r.remote_id == "peerA"  # remote id is still recorded
    assert calls == []  # but no SDP work happens


def test_handle_offer_sets_remote_and_answers(monkeypatch):
    r = make_receiver()
    calls = []
    monkeypatch.setattr(r, "_set_remote_sdp", lambda sdp, is_offer: calls.append(("set", sdp, is_offer)))
    monkeypatch.setattr(r, "_create_and_send_answer", lambda: calls.append(("answer",)))
    r._handle_offer(Envelope(type="offer", from_="peerA", data={"sdp": "v=0..."}))
    assert r.remote_id == "peerA"
    assert calls == [("set", "v=0...", True), ("answer",)]


def test_handle_offer_flushes_pending_remote_ice(monkeypatch):
    r = make_receiver()
    monkeypatch.setattr(r, "_set_remote_sdp", lambda *a, **k: None)
    monkeypatch.setattr(r, "_create_and_send_answer", lambda: None)
    r._pending_remote_ice = [(0, "c0"), (1, "c1")]
    r._handle_offer(Envelope(type="offer", from_="peerA", data={"sdp": "v=0..."}))
    assert r.webrtc.emitted == [
        ("add-ice-candidate", (0, "c0")),
        ("add-ice-candidate", (1, "c1")),
    ]
    assert r._pending_remote_ice == []


# --------------------------- local candidate ----------------------------

def test_local_candidate_buffered_until_remote_known():
    r = make_receiver()
    r.remote_id = None
    r._on_ice_candidate(None, 1, "local-cand")
    assert r._pending_local_ice == [(1, "local-cand")]
    assert r.signal.calls == []


def test_local_candidate_sent_when_remote_known():
    r = make_receiver()
    r.remote_id = "peerA"
    r._on_ice_candidate(None, 1, "local-cand")
    assert r._pending_local_ice == []
    assert r.signal.calls == [("candidate", "peerA", 1, "local-cand")]


def test_send_local_ice_noop_without_remote_or_signal():
    r = make_receiver()
    r.remote_id = None
    r._send_local_ice(0, "c")
    assert r.signal.calls == []
