IPC Two Procs via Flask (E2E)

What this verifies
- Flask server: health, serving base64 payload, graceful shutdown.
- posix_ipc: named semaphore mutual exclusion across writer/reader.
- CUDA IPC: cross‑process visibility of tensor writes when CUDA is available.
- Base64 + pickle: plain stdlib `pickle.loads(base64.b64decode(...))` works.

How to run
- pixi run bash tests/e2e/ipc_two_procs_flask/run.sh

Artifacts
- Logs under tests/e2e/ipc_two_procs_flask/logs/<timestamp>/
  - writer.log: server lifecycle, semaphore name, writes/observations.
  - reader.log: healthz/GET base64, plain decode check, read/write means.
  - summary.txt: PASS | SKIP_NO_CUDA.

Notes
- No stdin/stdout is used to pass control. Writer writes v1, reader reads it
  and then writes v2. Writer polls and observes v2, then shuts the server down
  gracefully.
- On CPU‑only runs the CUDA checks are skipped but posix_ipc and base64 paths
  are still exercised.

