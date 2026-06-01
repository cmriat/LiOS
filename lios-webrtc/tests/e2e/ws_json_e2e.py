import json
import logging
import time

from gst_webrtc.services import JSONWebSocketAPIServer, spawn_json_ws_client_process


def wait_port(server: JSONWebSocketAPIServer, timeout: float = 3.0) -> int:
    # Wait for actual bound port when port=0
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = server.bound_port()
        if p != 0:
            return p
        time.sleep(0.01)
    raise TimeoutError("server did not bind to a port in time")


def drain_until(queue, pred, timeout: float = 5.0):
    # Pull items until predicate returns True or timeout
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            obj = queue.get(timeout=0.2)
        except Exception:
            continue
        if pred(obj):
            return obj
    raise TimeoutError("condition not met in time")


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    # Start server with short heartbeat for quick E2E
    srv = JSONWebSocketAPIServer(host="127.0.0.1", port=0, heartbeat_interval=0.5)
    srv.start()
    port = wait_port(srv)
    print(f"[server] listening on ws://127.0.0.1:{port}")

    # Start client in a separate process
    proc, to_server, from_server = spawn_json_ws_client_process(host="127.0.0.1", port=port, deliver_heartbeats=True)
    print("[client] process started")

    # Wait for client to establish /ws/to-client by receiving heartbeat first
    hb0 = drain_until(from_server, lambda o: isinstance(o, dict) and o.get("type") == "ka", timeout=10.0)
    print("[ok] initial heartbeat received:", json.dumps(hb0))

    # Test: server -> client
    payload_out = {"dir": "server->client", "n": 1}
    srv.send_json(payload_out)
    got = drain_until(from_server, lambda o: isinstance(o, dict) and o.get("dir") == "server->client")
    assert got == payload_out, f"mismatch: {got}"
    print("[ok] server->client JSON delivered")

    # Test: client -> server
    payload_in = {"dir": "client->server", "n": 2}
    to_server.put(payload_in)
    got2 = srv.recv_json(timeout=5.0)
    assert got2 == payload_in, f"mismatch: {got2}"
    print("[ok] client->server JSON delivered")

    # Test: heartbeat from server (application-level)
    hb = drain_until(from_server, lambda o: isinstance(o, dict) and o.get("type") == "ka", timeout=5.0)
    print("[ok] received heartbeat:", json.dumps(hb))

    # Cleanup
    proc.terminate()
    proc.join(timeout=2.0)
    srv.stop()
    print("[done] E2E test finished successfully")


if __name__ == "__main__":
    main()
