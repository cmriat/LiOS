# Adapted from Physical-Intelligence/openpi (Apache-2.0). See NOTICE for details.
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Optional

# NOTE: Keep comments minimal and in English as requested.

log = logging.getLogger(__name__)


class LeakyQueue1:
    """Single-slot, overwrite-on-put, non-consuming peek semantics.

    - put_nowait(x): replace the current item; signal waiters.
    - get()/peek(): return latest item without consuming it; optionally block.
    """

    _SENTINEL = object()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._item: Any = self._SENTINEL

    def put_nowait(self, item: Any) -> None:
        with self._lock:
            self._item = item
            self._cond.notify_all()

    def peek(self, *, timeout: Optional[float] = None) -> Any:
        with self._lock:
            if self._item is not self._SENTINEL:
                return self._item
            if timeout is None:
                while self._item is self._SENTINEL:
                    self._cond.wait()
                return self._item
            end = time.monotonic() + float(timeout)
            while self._item is self._SENTINEL:
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise Empty()
                self._cond.wait(timeout=remaining)
            return self._item

    # Alias for compatibility with Queue API expectations
    def get(self, timeout: Optional[float] = None) -> Any:  # type: ignore[override]
        return self.peek(timeout=timeout)

    def get_nowait(self) -> Any:  # type: ignore[override]
        return self.peek(timeout=0)

    def qsize(self) -> int:
        with self._lock:
            return 0 if self._item is self._SENTINEL else 1

    def empty(self) -> bool:
        return self.qsize() == 0

    def full(self) -> bool:
        return False


@dataclass
class JSONQueues:
    """Thread-safe queues bridging external threads and WS handlers."""

    to_client: Queue  # actions → client
    from_client: LeakyQueue1  # states ← client (leaky size=1)


class JSONWebSocketAPIServer:
    """Two-WS-endpoint JSON bridge with server-side keepalive.

    Endpoints (default paths):
      - /ws/from-client: client → server JSON states (recv only)
      - /ws/to-client:   server → client JSON actions (send only)

    Lifecycle: call start() to run in a background thread; stop() to shutdown.
    External threads interact via `queues.to_client` and `queues.from_client`.
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 18081,
        path_from_client: str = "/ws/from-client",
        path_to_client: str = "/ws/to-client",
        max_queue_size: int = 1024,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        heartbeat_interval: float = 30.0,
        queues: Optional[JSONQueues] = None,
    ) -> None:
        # Queues visible to external threads
        self.queues = (
            queues if queues is not None else JSONQueues(to_client=Queue(max_queue_size), from_client=LeakyQueue1())
        )

        # Server config
        self.host = host
        self.port = port
        self.path_from_client = path_from_client
        self.path_to_client = path_to_client
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._heartbeat_interval = heartbeat_interval

        # Thread/loop state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._server: Optional[asyncio.AbstractServer] = None

        # Track one active connection per endpoint
        self._ws_from_client = None
        self._ws_to_client = None

        # No hard dependency on inference buffer; pure queue bridge.

    # -------- External API --------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ws-json-api", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._loop or not self._thread or not self._stop_event:
            return
        fut = asyncio.run_coroutine_threadsafe(self._shutdown_async(), self._loop)
        try:
            fut.result(timeout=3.0)
        except Exception as e:  # pragma: no cover
            log.warning("WS shutdown error: %s", e)
        self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None
        self._stop_event = None

    def bound_port(self) -> int:
        """Return the actual bound port (useful when `port=0`)."""
        server = self._server
        if server and server.sockets:
            try:
                return server.sockets[0].getsockname()[1]
            except Exception:  # pragma: no cover
                pass
        return self.port

    def send_json(self, obj: Any) -> None:
        """Enqueue a JSON-serializable object to the to-client stream (actions)."""
        try:
            self.queues.to_client.put_nowait(obj)
        except Full:
            # Drop oldest to keep the stream moving under backpressure.
            try:
                _ = self.queues.to_client.get_nowait()
            except Empty:  # pragma: no cover
                pass
            self.queues.to_client.put_nowait(obj)

    def recv_json(self, *, timeout: Optional[float] = None) -> Any:
        """Blocking read for a JSON object received from client (states, non-consuming)."""
        return self.queues.from_client.get(timeout=timeout)

    # Aliases to make intent explicit for callers
    # action → client (producer pushes actions here)
    def send_action(self, action: Any) -> None:
        self.send_json(action)

    # state ← client (consumer pulls states from here)
    def recv_state(self, *, timeout: Optional[float] = None) -> Any:
        return self.recv_json(timeout=timeout)

    @property
    def action_out_queue(self) -> Queue:
        return self.queues.to_client

    @property
    def state_in_queue(self) -> Queue:
        return self.queues.from_client

    # -------- Internal: thread & loop --------
    def _run(self) -> None:
        # Isolated event loop per thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._stop_event = asyncio.Event()
        try:
            loop.run_until_complete(self._serve())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _serve(self) -> None:
        import websockets

        # Single handler dispatching by path
        async def handler(*args) -> None:
            # Compat: websockets>=12 passes (ws,), older passes (ws, path)
            if len(args) == 1:
                ws = args[0]
                path = getattr(ws, "path", None)
                if not path:
                    req = getattr(ws, "request", None)
                    path = getattr(req, "path", "")
            else:
                ws, path = args[0], args[1]
            log.info("WS request path: %s", path)
            if path == self.path_from_client:
                await self._handle_from_client(ws)
                return
            if path == self.path_to_client:
                await self._handle_to_client(ws)
                return
            await ws.close(code=1008, reason="invalid path")

        # Start server with built-in ping/pong keepalive
        server = await websockets.serve(
            handler,
            host=self.host,
            port=self.port,
            ping_interval=self._ping_interval,
            ping_timeout=self._ping_timeout,
            max_size=None,  # JSON payload size not limited here
        )
        self._server = server
        log.info(
            "WS JSON API serving on ws://%s:%d (in=%s, out=%s)",
            self.host,
            self.port,
            self.path_from_client,
            self.path_to_client,
        )

        try:
            assert self._stop_event is not None
            await self._stop_event.wait()
        finally:
            server.close()
            await server.wait_closed()

    async def _shutdown_async(self) -> None:
        if self._stop_event:
            self._stop_event.set()

    # -------- Handlers --------
    async def _handle_from_client(self, ws) -> None:
        # Accepts client → server JSON and enqueues for external consumers.
        self._ws_from_client = ws
        log.info("WS from-client connected")
        try:
            async for raw in ws:
                try:
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8", "ignore")
                    obj = json.loads(raw)
                except Exception:
                    # Strict JSON only
                    continue
                # Leaky semantics: overwrite latest state
                self.queues.from_client.put_nowait(obj)
        except Exception as e:  # pragma: no cover - connection lifecycle
            log.debug("WS from-client closed: %s", e)
        finally:
            self._ws_from_client = None
            log.info("WS from-client disconnected")

    async def _handle_to_client(self, ws) -> None:
        # Sends server → client JSON actions from queue; emits periodic heartbeat JSON.
        self._ws_to_client = ws
        log.info("WS to-client connected")
        last_hb = time.monotonic()
        try:
            while True:
                # Drain queue fast to reduce latency
                try:
                    item = self.queues.to_client.get_nowait()
                    await ws.send(json.dumps(item))
                    continue
                except Empty:
                    pass

                # Heartbeat message (application-level) to keep proxies alive
                now = time.monotonic()
                if self._heartbeat_interval and (now - last_hb) >= self._heartbeat_interval:
                    await ws.send(json.dumps({"type": "ka", "ts": time.time()}))
                    last_hb = now

                # Check stop signal periodically
                if self._stop_event and self._stop_event.is_set():
                    break

                await asyncio.sleep(0.001)  # 1ms
        except Exception as e:  # pragma: no cover - connection lifecycle
            log.debug("WS to-client closed: %s", e)
        finally:
            self._ws_to_client = None
            log.info("WS to-client disconnected")

    # No buffer-binding logic: keep this service a pure WS↔queue bridge.
