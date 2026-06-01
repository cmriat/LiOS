# E2E Smoke Test

Run an end-to-end smoke test locally using pixi and the built-in Go signaling server.

Quick start:

```
pixi install
./tests/e2e/smoke.sh
```

What it does:
- Starts `signal-server` via `go run . serve --addr :18080`.
- Runs `receiver.py` and `sender_sw.py` (software x264enc) headlessly via `pixi run`.
- Watches logs for successful WebRTC link/answer.
- Writes a JSON report to `tests/e2e/report.json` and exits non-zero on failure.

Artifacts:
- Logs: `tests/e2e/logs/{signal,receiver,sender}.log`
- Report: `tests/e2e/report.json`

Environment:
- `TIMEOUT_SEC` (optional): overall wait timeout (default 25).

---

# GPU Sink E2E

This test validates `gpu_sink_save.py` end-to-end by saving PNG frames via PIL.

Run:

```
pixi install
./tests/e2e/gpu_sink.sh
```

What it does:
- Starts the Go signaling server on `:18080`.
- Runs `gpu_sink_save.py` as the receiver to pull NumPy frames and save PNGs.
- Runs `sender_sw.py` to produce an H264 RTP stream.
- Waits until the requested number of frames are saved, then verifies images using PIL.
- Writes a JSON report to `tests/e2e/gpu_sink_report.json`.

Artifacts:
- Saved frames: `tests/e2e/artifacts/gpu_frames/*.png`
- Logs: `tests/e2e/logs/{signal_gpu,gpu_sink_receiver,gpu_sink_sender}.log`
- Report: `tests/e2e/gpu_sink_report.json`

Environment:
- `FRAMES` (optional): number of frames to save (default 5)
- `PORT` (optional): signal server port (default 18080)
- `SIGNAL_URL` (optional): override WebSocket URL
- `TIMEOUT_SEC` (optional): overall timeout (default 40)
