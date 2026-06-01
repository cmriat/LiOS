import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional

import websockets


# Mirrors the Envelope in signal-server/server/server.go.
# `from` is renamed to `from_` to avoid clashing with the Python keyword.
@dataclass
class Envelope:
    type: str
    room: Optional[str] = None
    from_: Optional[str] = None
    to: Optional[str] = None
    data: Any = None
    text: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "room": self.room,
            "from": self.from_,
            "to": self.to,
            "data": self.data,
            "text": self.text,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @staticmethod
    def from_json(raw: str) -> "Envelope":
        m = json.loads(raw)
        data = m.get("data")
        # Tolerate servers that double-encode `data` as a JSON string.
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                pass
        return Envelope(
            type=m.get("type"),
            room=m.get("room"),
            from_=m.get("from"),
            to=m.get("to"),
            data=data,
            text=m.get("text"),
        )


class SignalClient:
    """
    Minimal WebSocket signaling client.

    - connect/close (supports `async with`)
    - join a room and broadcast `ready`
    - send/receive offer/answer/candidate
    - iterate incoming messages as `Envelope`
    - thread-safe send: callable from non-async threads (e.g. GStreamer callbacks)
    """

    def __init__(self, url: str, room: str, me: str):
        self.url = url
        self.room = room
        self.me = me
        self.ws: Optional[websockets.ClientProtocol] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # ---------- lifecycle ----------
    async def connect(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.ws = await websockets.connect(self.url)

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
            self.ws = None

    async def __aenter__(self) -> "SignalClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---------- send (thread-safe wrappers) ----------
    async def _send_env(self, env: Envelope) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        await self.ws.send(env.to_json())

    def send_env(self, env: Envelope):
        if not self.loop:
            raise RuntimeError("Event loop is not ready")
        return asyncio.run_coroutine_threadsafe(self._send_env(env), self.loop)

    # ---------- common actions ----------
    def join(self, role: Optional[str] = None, extra: Optional[dict] = None):
        data = dict(extra or {})
        if role:
            data.setdefault("role", role)
        env = Envelope(type="join", room=self.room, from_=self.me, data=data)
        return self.send_env(env)

    def ready(self):
        env = Envelope(type="ready", room=self.room, from_=self.me)
        return self.send_env(env)

    def offer(self, to: str, sdp_text: str):
        env = Envelope(
            type="offer",
            room=self.room,
            from_=self.me,
            to=to,
            data={"type": "offer", "sdp": sdp_text},
        )
        return self.send_env(env)

    def answer(self, to: str, sdp_text: str):
        env = Envelope(
            type="answer",
            room=self.room,
            from_=self.me,
            to=to,
            data={"sdp": sdp_text},
        )
        return self.send_env(env)

    def candidate(self, to: str, mline_index: int, candidate: str):
        env = Envelope(
            type="candidate",
            room=self.room,
            from_=self.me,
            to=to,
            data={"candidate": candidate, "sdpMLineIndex": int(mline_index)},
        )
        return self.send_env(env)

    # ---------- receive ----------
    async def recv(self) -> Envelope:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        raw = await self.ws.recv()
        return Envelope.from_json(raw)

    async def messages(self) -> AsyncIterator[Envelope]:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        async for raw in self.ws:  # type: ignore[attr-defined]
            yield Envelope.from_json(raw)

    def __aiter__(self) -> AsyncIterator[Envelope]:
        return self.messages()

    # ---------- helper: peer discovery ----------
    async def discover_peer(self, timeout: Optional[float] = None) -> Optional[str]:
        """Wait for 'peers' or 'peer-join'; return the first peer id, or None on timeout."""

        async def _wait() -> Optional[str]:
            while True:
                m = await self.recv()
                if m.type == "peers":
                    lst: List[str] = m.data or []
                    if isinstance(lst, list) and lst:
                        return lst[0]
                elif m.type == "peer-join":
                    return m.from_
                # ignore other message types (e.g. ready / peer-leave)

        try:
            if timeout is None:
                return await _wait()
            return await asyncio.wait_for(_wait(), timeout)
        except asyncio.TimeoutError:
            return None
