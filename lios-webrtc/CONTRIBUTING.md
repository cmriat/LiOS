# Contributing to LiOS

Thanks for your interest in improving LiOS! This component is the GPU-direct
edge→cloud image-transport path of the LiOS embodied-AI infrastructure. The notes
below get you to a working dev setup and a green CI.

## Development environment

The runtime (CUDA 12.9, PyTorch, GStreamer 1.26, `posix_ipc`, pytest, ruff, …) is
managed with [pixi](https://pixi.sh). **Run every command through `pixi run`** — do
not use a system `python` / `pip`.

```bash
pixi install                      # resolve and create the environment
pixi run python -c "import gi"    # sanity check
```

The Go signaling server lives under `signal-server/` and builds independently:

```bash
cd signal-server && go build -o webrtcssvr .
```

See [`README.md`](README.md) for the repository layout and a runnable quick start,
and [`design.md`](design.md) for the component architecture.

## Linting

```bash
pixi run ruff check src tests
```

CI runs the same check. Fix all findings (or justify an ignore in `pyproject.toml`)
before opening a PR.

## Tests

```bash
pixi run pytest                   # GPU tests are skipped by default
pixi run pytest -m gpu            # only the GPU-marked tests (needs a CUDA GPU)
```

When you add tests, **do not modify the code under test to make a test pass** —
tests should pin existing behavior or drive a deliberate, separately-reviewed change.
Tests use `importorskip` so they degrade gracefully when optional deps (`gi`,
`torch`, `posix_ipc`) are missing.

## Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/); the CHANGELOG is
generated from them with [git-cliff](https://git-cliff.org/). Examples:

```
feat(receiver): add trickle-ICE restart handling
fix(gpu_sink): correct RGBA stride padding in _map_to_numpy
docs: clarify near-net relay deployment in README
```

## Pull requests

1. Fork and branch from `main`.
2. Make sure the three CI jobs pass locally: `ruff check`, `pytest -m "not gpu"`, and
   `go build` for the signaling server (if you touched Go).
3. Update docs (`README.md` / `README.zh-CN.md` / `design.md` / `docs/`) when behavior
   or interfaces change.
4. Fill in the pull-request template and link any related issue.

## License

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).
