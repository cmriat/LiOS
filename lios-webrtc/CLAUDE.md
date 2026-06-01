# LiOS — Claude 工作约定

## 环境：一律用 `pixi run`

本仓库的运行环境由 **pixi** 管理（见 `pixi.toml` / `pixi.lock`，包含 CUDA 12.9、
PyTorch、GStreamer 1.26、posix_ipc、pytest、ruff 等）。

**任何命令都必须用 `pixi run` 前缀在 pixi 环境内执行，不要用系统的 `python` / `pip`。**

常用命令：

- 跑脚本：`pixi run python path/to/script.py`
- 跑测试：`pixi run pytest`（GPU 用例默认跳过；只跑 GPU 用例：`pixi run pytest -m gpu`）
- 跑 lint：`pixi run ruff check src tests`
- 起信令服务：`cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080`
- LiveKit 基准（独立环境）：`pixi run -e livekit python benchmark/throughput/livekit_subscriber.py`

## 其它

- 始终用简体中文回复。
- 写测试时不要修改被测代码。
