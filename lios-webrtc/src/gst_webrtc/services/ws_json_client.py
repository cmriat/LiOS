from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

try:
    import multiprocessing as mp
except Exception:  # pragma: no cover
    mp = None  # type: ignore

# Minimal, English-only comments per repo guidance.

log = logging.getLogger(__name__)


@dataclass
class ClientQueues:
    """Queues for client external interface."""

    to_server: Any  # queue-like: put_nowait/get
    from_server: Any


class JSONWebSocketClient:
    """Client for paired WS JSON endpoints with keepalive and reconnection."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 5002,
        path_from_client: str = "/ws/from-client",
        path_to_client: str = "/ws/to-client",
        queues: Optional[ClientQueues] = None,
        max_queue_size: int = 1024,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        deliver_heartbeats: bool = True,
        upstream_heartbeat_interval: float = 0.0,
        reconnect_initial: float = 1.0,
        reconnect_max: float = 10.0,
    ) -> None:
        from queue import Queue

        self.host = host
        self.port = port
        self.path_from_client = path_from_client
        self.path_to_client = path_to_client
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.deliver_heartbeats = deliver_heartbeats
        self.upstream_heartbeat_interval = upstream_heartbeat_interval
        self.reconnect_initial = reconnect_initial
        self.reconnect_max = reconnect_max

        if queues is None:
            queues = ClientQueues(to_server=Queue(max_queue_size), from_server=Queue(max_queue_size))
        self.queues = queues

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # --- external API ---
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run_loop, name="ws-json-client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self._thread = None
        self._loop = None

    # --- internals ---
    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._main())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    async def _main(self) -> None:
        recv_task = asyncio.create_task(self._recv_loop())
        send_task = asyncio.create_task(self._send_loop())
        try:
            await asyncio.wait([recv_task, send_task], return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in (recv_task, send_task):
                t.cancel()
                with contextlib.suppress(Exception):
                    await t

    async def _recv_loop(self) -> None:
        import websockets

        backoff = self.reconnect_initial
        url = f"ws://{self.host}:{self.port}{self.path_to_client}"
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    max_size=None,
                ) as ws:
                    log.info("Client recv connected: %s", url)
                    backoff = self.reconnect_initial
                    async for raw in ws:
                        try:
                            obj = json.loads(raw)
                        except Exception:
                            continue
                        if not self.deliver_heartbeats and isinstance(obj, dict) and obj.get("type") == "ka":
                            continue
                        self._queue_put(self.queues.from_server, obj)
            except Exception as e:
                log.debug("Client recv loop error: %s", e)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(self.reconnect_max, backoff * 2)

    async def _send_loop(self) -> None:
        import websockets

        backoff = self.reconnect_initial
        url = f"ws://{self.host}:{self.port}{self.path_from_client}"
        last_hb = time.monotonic()
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout,
                    max_size=None,
                ) as ws:
                    log.info("Client send connected: %s", url)
                    backoff = self.reconnect_initial
                    last_hb = time.monotonic()
                    while not self._stop.is_set():
                        item = self._queue_get_nowait(self.queues.to_server)
                        if item is not _QueueEmpty:
                            await ws.send(json.dumps(item))
                            continue
                        if (
                            self.upstream_heartbeat_interval
                            and (time.monotonic() - last_hb) >= self.upstream_heartbeat_interval
                        ):
                            await ws.send(json.dumps({"type": "ka", "ts": time.time()}))
                            last_hb = time.monotonic()
                        await asyncio.sleep(0.05)
            except Exception as e:
                log.debug("Client send loop error: %s", e)
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(self.reconnect_max, backoff * 2)

    @staticmethod
    def _queue_put(q, obj):
        from queue import Full

        try:
            q.put_nowait(obj)
        except Full:
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            q.put_nowait(obj)

    @staticmethod
    def _queue_get_nowait(q):
        try:
            return q.get_nowait()
        except Exception:
            return _QueueEmpty


_QueueEmpty = object()


def _client_proc_entry(
    host: str,
    port: int,
    path_from_client: str,
    path_to_client: str,
    to_server,
    from_server,
    ping_interval: float,
    ping_timeout: float,
    deliver_heartbeats: bool,
    upstream_heartbeat_interval: float,
    reconnect_initial: float,
    reconnect_max: float,
):
    # SIGTERM for graceful exit in containerized runs
    def _term_handler(signum, frame):  # noqa: ARG001
        os._exit(0)

    signal.signal(signal.SIGTERM, _term_handler)
    client = JSONWebSocketClient(
        host=host,
        port=port,
        path_from_client=path_from_client,
        path_to_client=path_to_client,
        queues=ClientQueues(to_server=to_server, from_server=from_server),
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        deliver_heartbeats=deliver_heartbeats,
        upstream_heartbeat_interval=upstream_heartbeat_interval,
        reconnect_initial=reconnect_initial,
        reconnect_max=reconnect_max,
    )
    client.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        client.stop()


# --- process wrapper ---
def spawn_json_ws_client_process(
    *,
    host: str = "127.0.0.1",
    port: int = 5002,
    path_from_client: str = "/ws/from-client",
    path_to_client: str = "/ws/to-client",
    max_queue_size: int = 1024,
    ping_interval: float = 20.0,
    ping_timeout: float = 20.0,
    deliver_heartbeats: bool = True,
    upstream_heartbeat_interval: float = 0.0,
    reconnect_initial: float = 1.0,
    reconnect_max: float = 10.0,
):
    """Start client in a separate process and return (proc, to_server, from_server)."""
    if mp is None:  # pragma: no cover
        raise RuntimeError("multiprocessing is unavailable")

    ctx = mp.get_context("spawn")
    to_server = ctx.Queue(max_queue_size)
    from_server = ctx.Queue(max_queue_size)

    proc = ctx.Process(
        target=_client_proc_entry,
        name="ws-json-client-proc",
        daemon=True,
        args=(
            host,
            port,
            path_from_client,
            path_to_client,
            to_server,
            from_server,
            ping_interval,
            ping_timeout,
            deliver_heartbeats,
            upstream_heartbeat_interval,
            reconnect_initial,
            reconnect_max,
        ),
    )
    proc.start()
    return proc, to_server, from_server


# Optional quick demo
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    p, to_srv, from_srv = spawn_json_ws_client_process()
    try:
        to_srv.put({"hello": "server"})
        t0 = time.time()
        while time.time() - t0 < 5.0:
            try:
                msg = from_srv.get(timeout=1.0)
                print("recv:", msg)
            except Exception:
                pass
    finally:
        p.terminate()
        p.join(timeout=2.0)
