from __future__ import annotations

from dataclasses import dataclass

from .flask_api import APIServer, InferenceBufferProvider
from .ws_json_api import JSONWebSocketAPIServer


@dataclass
class ServiceSuite:
    """Convenience launcher to start Flask and WS JSON APIs together."""

    http: APIServer
    ws: JSONWebSocketAPIServer

    @classmethod
    def create(
        cls,
        *,
        host_http: str = "127.0.0.1",
        port_http: int = 5001,
        host_ws: str = "127.0.0.1",
        port_ws: int = 5002,
        provider: InferenceBufferProvider | None = None,
    ) -> "ServiceSuite":
        provider = provider or InferenceBufferProvider()
        http = APIServer(provider, host=host_http, port=port_http)
        ws = JSONWebSocketAPIServer(host=host_ws, port=port_ws, provider=provider)
        return cls(http=http, ws=ws)

    def start(self) -> None:
        self.http.start()
        self.ws.start()

    def stop(self) -> None:
        self.ws.stop()
        self.http.stop()
