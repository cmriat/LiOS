"""Pure-logic unit tests for WebRTCSender negotiation / ICE state machine.

No GPU / no network / no signaling server: a bare instance is created via
__new__ and collaborators are faked. Skips where `gi` is unavailable.
"""

import time as _time_mod
import types

import pytest

pytest.importorskip("gi")

from _fakes import FakeBin, FakePad, FakeSignal, FakeState, FakeWebRTC  # noqa: E402

from gst_webrtc import init_gst  # noqa: E402
from gst_webrtc.sender.core import WebRTCSender  # noqa: E402

init_gst()


def make_sender() -> WebRTCSender:
    s = object.__new__(WebRTCSender)
    s.signal = FakeSignal()
    s.webrtc = FakeWebRTC()
    s._remote_id = "peerB"
    s._making_offer = False
    s._rebuilding = False
    s._sources = {}
    s._ice_disconnected_at = None
    s._ice_restart_deadline = None
    s._last_remote_left_ts = None
    return s


def _handle(bin_):
    return types.SimpleNamespace(bin=bin_)


# --------------------------- linked sources -----------------------------

def test_has_linked_sources_empty():
    assert make_sender()._has_linked_sources() is False


def test_has_linked_sources_true_when_pad_linked():
    s = make_sender()
    s._sources = {"a": _handle(FakeBin(FakePad(linked=True)))}
    assert s._has_linked_sources() is True


def test_has_linked_sources_false_when_unlinked_or_missing():
    s = make_sender()
    s._sources = {"a": _handle(FakeBin(FakePad(linked=False))), "b": _handle(FakeBin(None))}
    assert s._has_linked_sources() is False


# ------------------------------ _send_offer -----------------------------

@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: setattr(s, "signal", None),
        lambda s: setattr(s, "_remote_id", None),
        lambda s: setattr(s, "_making_offer", True),
        lambda s: setattr(s, "_rebuilding", True),
    ],
)
def test_send_offer_guard_blocks_emit(mutate):
    s = make_sender()
    mutate(s)
    before = s._making_offer
    s._send_offer(label="offer")
    assert s.webrtc.emitted == []
    assert s._making_offer == before  # guard must not flip the flag


def test_send_offer_emits_create_offer_and_sets_flag():
    s = make_sender()
    s._send_offer(label="offer")
    assert "create-offer" in s.webrtc.emitted_names
    assert s._making_offer is True  # stays set until the (async) reply callback


# --------------------------- negotiate routing --------------------------

def test_maybe_negotiate_guard_blocks(monkeypatch):
    s = make_sender()
    s.signal = None
    sent = []
    monkeypatch.setattr(s, "_send_offer", lambda *a, **k: sent.append(1))
    s._maybe_negotiate()
    assert sent == []


def test_maybe_negotiate_sends_when_linked(monkeypatch):
    s = make_sender()
    monkeypatch.setattr(s, "_has_linked_sources", lambda: True)
    sent = []
    monkeypatch.setattr(s, "_send_offer", lambda *a, **k: sent.append(1))
    s._maybe_negotiate()
    assert sent == [1]


def test_maybe_negotiate_skips_when_no_linked_sources(monkeypatch):
    s = make_sender()
    monkeypatch.setattr(s, "_has_linked_sources", lambda: False)
    sent = []
    monkeypatch.setattr(s, "_send_offer", lambda *a, **k: sent.append(1))
    s._maybe_negotiate()
    assert sent == []


def test_on_negotiation_needed_respects_linked_sources(monkeypatch):
    s = make_sender()
    sent = []
    monkeypatch.setattr(s, "_send_offer", lambda *a, **k: sent.append(1))
    monkeypatch.setattr(s, "_has_linked_sources", lambda: False)
    s._on_negotiation_needed()
    assert sent == []
    monkeypatch.setattr(s, "_has_linked_sources", lambda: True)
    s._on_negotiation_needed()
    assert sent == [1]


# ----------------------------- local ICE out ----------------------------

def test_on_ice_candidate_sends_when_ready():
    s = make_sender()
    s._on_ice_candidate(None, 1, "cand")
    assert s.signal.calls == [("candidate", "peerB", 1, "cand")]


def test_on_ice_candidate_noop_without_remote():
    s = make_sender()
    s._remote_id = None
    s._on_ice_candidate(None, 1, "cand")
    assert s.signal.calls == []


# --------------------------- ICE state machine --------------------------

def test_ice_state_disconnected_records_timestamp(monkeypatch):
    s = make_sender()
    monkeypatch.setattr(_time_mod, "time", lambda: 1000.0)
    s.webrtc.set_property("ice-connection-state", FakeState("disconnected"))
    s._on_ice_state(None, None)
    assert s._ice_disconnected_at == 1000.0


def test_ice_state_connected_clears_timers():
    s = make_sender()
    s._ice_disconnected_at = 5.0
    s._ice_restart_deadline = 9.0
    s.webrtc.set_property("ice-connection-state", FakeState("connected"))
    s._on_ice_state(None, None)
    assert s._ice_disconnected_at is None
    assert s._ice_restart_deadline is None


def test_ice_state_failed_triggers_restart(monkeypatch):
    s = make_sender()
    fired = []
    monkeypatch.setattr(s, "_restart_ice", lambda: fired.append(1))
    s.webrtc.set_property("ice-connection-state", FakeState("failed"))
    s._on_ice_state(None, None)
    assert fired == [1]


def test_ice_state_prolonged_disconnect_triggers_restart(monkeypatch):
    s = make_sender()
    s._ice_disconnected_at = 1000.0
    monkeypatch.setattr(_time_mod, "time", lambda: 1002.0)  # >1s later
    fired = []
    monkeypatch.setattr(s, "_restart_ice", lambda: fired.append(1))
    s.webrtc.set_property("ice-connection-state", FakeState("disconnected"))
    s._on_ice_state(None, None)
    assert fired == [1]


# ------------------------------ restart / tick --------------------------

def test_restart_ice_guarded_while_making_offer(monkeypatch):
    s = make_sender()
    s._making_offer = True
    sent = []
    monkeypatch.setattr(s, "_send_offer", lambda *a, **k: sent.append(a))
    s._restart_ice()
    assert sent == []
    assert s._ice_restart_deadline is None


def test_restart_ice_sends_offer_and_arms_deadline(monkeypatch):
    s = make_sender()
    monkeypatch.setattr(s, "_ice_restart_options", lambda: "OPTS")
    monkeypatch.setattr(_time_mod, "time", lambda: 2000.0)
    sent = []
    monkeypatch.setattr(s, "_send_offer", lambda options=None, **k: sent.append(options))
    s._restart_ice()
    assert sent == ["OPTS"]
    assert s._ice_restart_deadline == 2005.0


def test_tick_escalates_to_full_reset_when_deadline_passed(monkeypatch):
    s = make_sender()
    s._ice_restart_deadline = 100.0
    s.webrtc.set_property("connection-state", FakeState("failed"))
    monkeypatch.setattr(_time_mod, "time", lambda: 200.0)
    reset = []
    monkeypatch.setattr(s, "_full_reset_webrtc", lambda: reset.append(1))
    s.tick()
    assert reset == [1]
    assert s._ice_restart_deadline is None


def test_tick_noop_when_no_deadline(monkeypatch):
    s = make_sender()
    s._ice_restart_deadline = None
    reset = []
    monkeypatch.setattr(s, "_full_reset_webrtc", lambda: reset.append(1))
    s.tick()
    assert reset == []


def test_tick_noop_when_connected(monkeypatch):
    s = make_sender()
    s._ice_restart_deadline = 100.0
    s.webrtc.set_property("connection-state", FakeState("connected"))
    monkeypatch.setattr(_time_mod, "time", lambda: 200.0)
    reset = []
    monkeypatch.setattr(s, "_full_reset_webrtc", lambda: reset.append(1))
    s.tick()
    assert reset == []
