"""Lightweight fakes for sender/receiver pure-logic unit tests.

These have no GStreamer/GPU/network dependency: they stand in for `webrtcbin`,
the signaling client, and the few GStreamer objects the decision logic touches,
recording calls so tests can assert behavior.
"""

from __future__ import annotations


class FakeWebRTC:
    """Stand-in for webrtcbin: records emit() calls; dict-backed properties."""

    def __init__(self) -> None:
        self.emitted: list[tuple] = []
        self.props: dict = {}

    def emit(self, name, *args):
        self.emitted.append((name, args))
        return None

    def set_property(self, key, val):
        self.props[key] = val

    def get_property(self, key):
        return self.props.get(key)

    def connect(self, *_a, **_k):
        return 0

    @property
    def emitted_names(self) -> list[str]:
        return [n for n, _ in self.emitted]


class FakeSignal:
    """Stand-in for SignalClient: records signaling actions."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def offer(self, to, sdp):
        self.calls.append(("offer", to, sdp))

    def answer(self, to, sdp):
        self.calls.append(("answer", to, sdp))

    def candidate(self, to, mline, cand):
        self.calls.append(("candidate", to, mline, cand))

    def join(self, **k):
        self.calls.append(("join", k))

    def ready(self):
        self.calls.append(("ready",))

    @property
    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class FakeState:
    """Stand-in for a GStreamer state enum exposing `.value_nick`."""

    def __init__(self, nick: str) -> None:
        self.value_nick = nick


class FakePad:
    def __init__(self, linked: bool = True) -> None:
        self._linked = linked

    def is_linked(self) -> bool:
        return self._linked


class FakeBin:
    def __init__(self, src_pad: FakePad | None = None) -> None:
        self._src = src_pad

    def get_static_pad(self, _name):
        return self._src
