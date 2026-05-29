import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional

import websockets


# 与 signal-server/server/server.go 的 Envelope 对齐
# 仅用于 Python 侧封装收发，字段名保持一致（from -> from_ 避免关键字冲突）
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
        # 容错：部分服务可能把 data 再包了一层字符串
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
    简单的 WebSocket 信令客户端封装

    功能：
    - 建连/关闭（支持 async with）
    - 加入房间（join）与广播 ready
    - 统一封装 offer/answer/candidate 的收发
    - 提供迭代器按 Envelope 读取消息
    - 线程安全发送：可在 GStreamer 回调等非协程线程中调用
    """

    def __init__(self, url: str, room: str, me: str):
        self.url = url
        self.room = room
        self.me = me
        self.ws: Optional[websockets.ClientProtocol] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    # ---------- 基础生命周期 ----------
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

    # ---------- 发送（线程安全包装） ----------
    async def _send_env(self, env: Envelope) -> None:
        if not self.ws:
            raise RuntimeError("WebSocket is not connected")
        await self.ws.send(env.to_json())

    def send_env(self, env: Envelope):
        if not self.loop:
            raise RuntimeError("Event loop is not ready")
        return asyncio.run_coroutine_threadsafe(self._send_env(env), self.loop)

    # ---------- 常用动作 ----------
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

    # ---------- 接收 ----------
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

    # ---------- 辅助：发现对端 ----------
    async def discover_peer(self, timeout: Optional[float] = None) -> Optional[str]:
        """等待 'peers' 或 'peer-join'，返回第一个 peer id。超时返回 None。"""

        async def _wait() -> Optional[str]:
            while True:
                m = await self.recv()
                if m.type == "peers":
                    lst: List[str] = m.data or []
                    if isinstance(lst, list) and lst:
                        return lst[0]
                elif m.type == "peer-join":
                    return m.from_
                # 其他消息忽略（例如 ready / peer-leave 等）

        try:
            if timeout is None:
                return await _wait()
            return await asyncio.wait_for(_wait(), timeout)
        except asyncio.TimeoutError:
            return None
