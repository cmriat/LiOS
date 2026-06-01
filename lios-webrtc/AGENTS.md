# Repository Guidelines

- 用pixi run ... 跑任何命令，比如pixi run python test.py
- 如果这次的任务是写测试，永远不要修改被测代码  

## Project Structure & Module Organization
- `gst_webrtc/` — Python GStreamer WebRTC clients: `sender.py`, `receiver.py`, `receiver_autosink.py`; signaling helper in `ws_signal/`.
- `signal-server/` — Go WebRTC signaling server (Cobra CLI). Built binary: `signalsrv`.
- `benchmark/rtp_latency/` — RTP latency sender/receiver utilities.
- `dockerfiles/`, `pixi.toml`, `.envrc`, `justfile` — dev environment and helpers.
- `tests/` — place unit/e2e tests here (create if missing). Keep test code separate from production modules.

## Build, Test, and Development Commands
- Setup env: `pixi install` then `pixi shell` (or once: `direnv allow`).
- Start signaling: `cd signal-server && go build -o signalsrv . && ./signalsrv serve --addr :18080`.
- Send video: `ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws python gst_webrtc/sender.py`.
- Receive JPEG frames: `ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws FPS=1 FRAMES_DIR=./frames python gst_webrtc/receiver.py`.
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
- Override via env: `ROOM`, `SIGNAL_URL` (e.g., `ws://127.0.0.1:18080/ws`), `STUN`, `TURN`, `FPS`, `FRAMES_DIR`. Never commit secrets; update ignore lists as needed.
- GPU pipelines (e.g., `cudaupload`, `nvh264enc`) require appropriate drivers/plugins. Logging via `GST_DEBUG` (default `3`; raise to `4/5` when diagnosing pipelines).

## Agent‑Specific Instructions
- Scope: this file applies to the entire repository.
- Keep changes minimal and focused; avoid unrelated refactors.
- Follow the styles above; prefer small, reviewable PRs.
- Ask before destructive actions; document assumptions and commands used.

