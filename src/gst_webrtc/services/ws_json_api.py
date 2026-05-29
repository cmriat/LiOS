from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Optional

from gst_webrtc.inference_buffer_v2 import InferenceBufferV2

from .flask_api import InferenceBufferProvider

# NOTE: Keep comments minimal and in English as requested.

log = logging.getLogger(__name__)


@dataclass
class JSONQueues:
    """Thread-safe queues bridging external threads and WS handlers."""

    to_client: Queue  # items pushed by external threads to send to client
    from_client: Queue  # items received from client for external threads


class JSONWebSocketAPIServer:
    """
    Two-WS-endpoint JSON bridge with server-side keepalive.

    Endpoints (default paths):
      - /ws/from-client: client → server JSON (recv only)
      - /ws/to-client:   server → client JSON (send only)

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
        provider: "InferenceBufferProvider | None" = None,
        state_key: str = "arm_positions",
    ) -> None:
        # Queues visible to external threads
        self.queues = JSONQueues(to_client=Queue(max_queue_size), from_client=Queue(max_queue_size))

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

        # Shared inference buffer provider (optional)
        # Imported lazily to avoid tight coupling at import time.
        self._provider = provider
        self._state_key = state_key

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
        """Enqueue a JSON-serializable object to the to-client stream."""
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
        """Blocking read for a JSON object received from client."""
        return self.queues.from_client.get(timeout=timeout)

    # Convenience passthrough to share a live buffer instance
    def set_buffer(self, buf: "InferenceBufferV2") -> None:
        prov = self._provider
        if prov is None:
            # Import here to avoid hard dependency when not used
            from .flask_api import InferenceBufferProvider  # type: ignore

            prov = InferenceBufferProvider()
            self._provider = prov
        prov.set_buffer(buf)

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
                    obj = json.loads(raw)
                except Exception:
                    # Strict JSON only
                    continue
                try:
                    self.queues.from_client.put_nowait(obj)
                except Full:
                    try:
                        _ = self.queues.from_client.get_nowait()
                    except Empty:  # pragma: no cover
                        pass
                    self.queues.from_client.put_nowait(obj)

                # Best-effort state write into shared buffer
                try:
                    await self._maybe_write_state(obj)
                except Exception as e:  # keep WS alive on processing errors
                    log.debug("state write failed: %s", e)
        except Exception as e:  # pragma: no cover - connection lifecycle
            log.debug("WS from-client closed: %s", e)
        finally:
            self._ws_from_client = None
            log.info("WS from-client disconnected")

    async def _handle_to_client(self, ws) -> None:
        # Sends server → client JSON from queue; emits periodic heartbeat JSON.
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

                await asyncio.sleep(0.001) # 1ms
        except Exception as e:  # pragma: no cover - connection lifecycle
            log.debug("WS to-client closed: %s", e)
        finally:
            self._ws_to_client = None
            log.info("WS to-client disconnected")

    # -------- State processing --------
    async def _maybe_write_state(self, obj: Any) -> None:
        # Minimal validation of expected schema and provider availability
        prov = self._provider
        if prov is None:
            return
        try:
            from gst_webrtc.inference_buffer_v2 import InferenceBufferV2  # type: ignore
        except Exception:
            return
        buf = prov.get_buffer() if hasattr(prov, "get_buffer") else None
        if not isinstance(buf, InferenceBufferV2):
            return

        # Extract 7+7 positions; ignore velocities per request
        left = obj.get("left_arm", {}) if isinstance(obj, dict) else {}
        right = obj.get("right_arm", {}) if isinstance(obj, dict) else {}
        lp = left.get("positions", []) if isinstance(left, dict) else []
        rp = right.get("positions", []) if isinstance(right, dict) else []
        if not (isinstance(lp, list) and isinstance(rp, list) and len(lp) == 7 and len(rp) == 7):
            return

        # Compose into 14-dim float32 tensor; prefer existing device if any
        import torch  # local import to keep module import light

        key = self._state_key
        dst = buf.states.get(key)
        device = getattr(dst, "device", torch.device("cpu")) if hasattr(dst, "device") else torch.device("cpu")
        src = torch.tensor(list(lp) + list(rp), dtype=torch.float32, device=device)

        # Convert timestamp (ms→seconds) and store to metadata
        ts = obj.get("timestamp")
        try:
            tsf = float(ts)
            if tsf > 1e11:  # assume ms since epoch
                tsf = tsf / 1000.0
        except Exception:
            tsf = time.time()

        # Tight critical section on state lock
        if hasattr(buf, "hold_state_lock"):
            ctx = buf.hold_state_lock(timeout=None)
        else:
            # Fallback to main lock if state lock unavailable
            ctx = buf.hold_lock(timeout=None)
        with ctx:
            if dst is not None and tuple(dst.shape) == (14,) and dst.dtype == torch.float32:
                dst.copy_(src)
            else:
                buf.states[key] = src
            # metadata alias accepted; store unix seconds
            try:
                # use alias if available to match request wording
                if hasattr(buf, "metadata"):
                    buf.metadata["state_timestamp"] = tsf
                else:
                    buf.meta["state_timestamp"] = tsf
            except Exception:
                pass
