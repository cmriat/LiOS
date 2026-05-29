from __future__ import annotations

"""
Lightweight Flask API running in a background thread.

Purpose
-------
- Run an HTTP API on a non-main thread to expose data produced by the main
  thread (e.g., inference buffers) without blocking GStreamer pipelines.
- The API is extensible: new endpoints can be added to the Flask app/blueprint
  without changing the threading/server integration.

Key endpoint (initial)
----------------------
- GET /api/v1/inference/buffer/base64
  Returns a base64 string that can be decoded via:
    `pickle.loads(base64.b64decode(body))`

Usage
-----
from gst_webrtc.inference_buffer_v2 import InferenceBufferV2
from gst_webrtc.services.flask_api import InferenceBufferProvider, APIServer

provider = InferenceBufferProvider()
server = APIServer(provider, host="127.0.0.1", port=5001)
server.start()  # starts the Flask server in a background thread

# Somewhere in your main loop, update the provider with the latest buffer:
# provider.set_buffer(latest_buf)

# Shutdown when done:
# server.stop()
"""

import logging
import threading
from dataclasses import dataclass
from typing import Optional

from flask import Blueprint, Flask, Response, current_app, jsonify

try:
    # Optional import; only needed for type hints and packing
    from gst_webrtc.inference_buffer_v2 import InferenceBufferV2
except Exception:  # pragma: no cover - allow importing without torch in some envs
    InferenceBufferV2 = object  # type: ignore

try:
    # Werkzeug is bundled with Flask; used to run a WSGI server in a thread.
    from werkzeug.serving import make_server
except Exception as e:  # pragma: no cover
    raise RuntimeError("werkzeug is required to run the threaded API server") from e


log = logging.getLogger(__name__)


@dataclass
class InferenceBufferProvider:
    """
    Thread-safe holder for a live `InferenceBufferV2` instance.

    The main thread should call `set_buffer` whenever a new buffer is available.
    The Flask thread calls `get_base64` to expose the latest snapshot.
    """

    _lock: threading.RLock = threading.RLock()
    _buf: Optional[InferenceBufferV2] = None

    def set_buffer(self, buf: InferenceBufferV2) -> None:
        with self._lock:
            self._buf = buf

    def get_buffer(self) -> Optional[InferenceBufferV2]:
        with self._lock:
            return self._buf

    def get_base64(self) -> Optional[str]:
        with self._lock:
            if self._buf is None:
                return None
            # `InferenceBufferV2.pack_base64()` returns an ASCII base64 string
            return self._buf.pack_base64()  # type: ignore[attr-defined]

    def is_ready(self) -> bool:
        with self._lock:
            return self._buf is not None


def _create_app(provider: InferenceBufferProvider) -> Flask:
    app = Flask(__name__)
    app.config["provider"] = provider

    api = Blueprint("api", __name__, url_prefix="/api/v1")

    @api.get("/healthz")
    def healthz() -> Response:
        status = 200 if provider.is_ready() else 503
        return jsonify({"status": "ok" if status == 200 else "starting"}), status

    @api.get("/infer-buffer/base64")
    def get_inference_buffer_base64() -> Response:
        """
        Return the latest inference buffer as base64 pickled bytes.

        Body is a plain ASCII base64 string. Clients can reconstruct a plain
        dict via `pickle.loads(base64.b64decode(body))` without importing any
        repository modules. If no buffer is currently available, returns 404.
        """
        prov: InferenceBufferProvider = current_app.config["provider"]
        s = prov.get_base64()
        if s is None:
            return Response("", status=404, mimetype="text/plain; charset=utf-8")
        # Keep body as raw base64 string for simple client usage.
        return Response(s, mimetype="text/plain; charset=utf-8")

    app.register_blueprint(api)
    return app


class _ServerThread(threading.Thread):
    """
    Run a Werkzeug WSGI server in a dedicated thread.

    This avoids Flask's development reloader and provides a clean `shutdown()`.
    """

    def __init__(self, app: Flask, host: str, port: int) -> None:
        super().__init__(daemon=True)
        self._server = make_server(host, port, app)
        self._ctx = app.app_context()
        self._shutdown = threading.Event()

    def run(self) -> None:  # pragma: no cover - simple bridge to serve_forever
        log.info("Flask API server thread starting")
        self._ctx.push()
        try:
            self._server.serve_forever()
        finally:
            self._ctx.pop()
            log.info("Flask API server thread exited")

    def shutdown(self) -> None:
        self._server.shutdown()
        self._shutdown.set()


class APIServer:
    """
    Owns the Flask app + background server thread.

    - Call `start()` once to begin serving.
    - Call `stop()` to shutdown and join the thread.
    - Use `provider.set_buffer(...)` to update the latest buffer from the main
      thread at any cadence.
    """

    def __init__(self, provider: InferenceBufferProvider, *, host: str = "127.0.0.1", port: int = 5001) -> None:
        self.provider = provider
        self.host = host
        self.port = port
        self.app = _create_app(provider)
        self._thread: Optional[_ServerThread] = None

    # Convenience passthroughs for callers
    def set_buffer(self, buf: InferenceBufferV2) -> None:
        self.provider.set_buffer(buf)

    def is_ready(self) -> bool:
        return self.provider.is_ready()

    # Lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = _ServerThread(self.app, self.host, self.port)
        self._thread.start()
        log.info("Flask API serving on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._thread.shutdown()
        self._thread.join(timeout=3.0)
        self._thread = None

