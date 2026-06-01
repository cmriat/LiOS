# Repository Guidelines

- 用pixi run ... 跑任何命令，比如pixi run python test.py
- 如果这次的任务是写测试，永远不要修改被测代码  

## Project Structure & Module Organization
- `src/gst_webrtc/` — Python package: `sender/`, `receiver/`, `gpu_sink/` clients; `inference_buffer_v2.py` (CUDA-IPC buffer); `ws_signal/` signaling client; `services/` (Flask + WS JSON control API).
- `signal-server/` — Go WebRTC signaling server (Cobra CLI). Built binary: `webrtcssvr`.
- `examples/` — runnable two-camera sender / receiver (with inference buffer).
- `benchmark/` — throughput / latency benchmarks (`throughput/`, `livekit/`, `rtp_latency/`).
- `tests/` — unit and e2e tests. Keep test code separate from production modules.
- `dockerfiles/`, `pixi.toml`, `.envrc`, `justfile` — dev environment and helpers.

## Build, Test, and Development Commands
- Setup env: `pixi install` (run everything via `pixi run`; never use system `python`/`pip`).
- Start signaling: `cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080`.
- Run the sender (edge GPU): `pixi run python examples/two_camera_sender.py`.
- Run the receiver (cloud GPU): `pixi run python examples/two_camera_receiver_inferbuf.py --streams 2`.
- GStreamer smoke test: `pixi run test`.

## Coding Style & Naming Conventions
- Python: 4‑space indent, type hints when useful, `snake_case` for modules/functions, `PascalCase` for classes, prefer f‑strings. Keep logs concise.
- Go: format with `gofmt`, vet with `go vet`. Keep packages cohesive; add meaningful package/file headers.

## Testing Guidelines
- Frameworks: use `pytest` for unit tests; add e2e tests for complex flows.
- Location: put tests under `tests/` mirroring package paths. Do not embed tests in production files or modify code‑under‑test to accommodate tests.
- Naming: `test_*.py` and descriptive test names. Run with `pytest`.

## Commit & Pull Request Guidelines
- Use Conventional Commits (`feat:`, `fix:`, `chore:`). Subject ≤ 72 chars; body explains motivation and key changes.
- PRs include: purpose, run steps (commands + env vars), logs/screenshots if relevant, and any plugin/driver requirements (e.g., NVIDIA `nvh264enc`).

## Security & Configuration Tips
- Override via env: `ROOM`, `SIGNAL_URL` (e.g., `ws://127.0.0.1:18080/ws`), `STUN`, `TURN`, `VIDEO_SOURCE`, `CAMERAS`. Never commit secrets; update ignore lists as needed.
- GPU pipelines (e.g., `cudaupload`, `nvh264enc`) require appropriate drivers/plugins. Logging via `GST_DEBUG` (default `3`; raise to `4/5` when diagnosing pipelines).

## Agent‑Specific Instructions
- Scope: this file applies to the entire repository.
- Keep changes minimal and focused; avoid unrelated refactors.
- Follow the styles above; prefer small, reviewable PRs.
- Ask before destructive actions; document assumptions and commands used.

