# Repository Guidelines

## AI Agent Guidelines

- Run every command through `pixi run ...` (e.g. `pixi run python scripts/inference.py`,
  `pixi run -e dev pytest -q`, `pixi run -e dev torchrun ...`). Do not invoke `python`,
  `pytest`, or `torchrun` directly — that uses the host interpreter and bypasses the
  pinned environment.
- When the task is "write tests", never modify the code under test. Tests must pass
  or fail against the existing implementation. If a test surfaces a real bug, report
  it separately rather than silently fixing the function to make the test pass.

## 项目结构与模块组织
- `src/pi/`：核心库。
  - `models/`：PaliGemma / Gemma / SigLIP 的 transformers fork、`model.py`、`pi_config.py`、`tokenizer.py`。
  - `models_pytorch/`：Pi0 / Pi0.5 PyTorch 实现（`pi0_pytorch.py`、`gemma_pytorch.py`、`attention_pooling.py`、`preprocessing_pytorch.py`）。
  - `training/`：训练配置（`config.py`、`instance_config.py`、`optimizer.py`）。
  - `shared/`：`normalize`、`download`、`image_tools`。
  - `services/`：推理时的 WebSocket JSON API。
  - 顶层 `data.py`、`transforms.py`、`ema_model.py`、`inference_buffer_v2.py`。
- `scripts/`：可执行脚本。
  - `train/`：FSDP 训练入口、数据加载、norm stats 计算、profiling。
  - `deployment/`：WebRTC + WebSocket 部署相关 notebook 与协议说明。
  - `inference.py`：离线 single-shot 推理 CLI。
- `assets/`：模型与数据相关资源（勿提交私有权重）。
- `tests/`：pytest 测试用例（CPU 即可，详见下文「测试规范」）。
- `docs/`：双语主题文档（架构、训练、数据、部署、移植）。

## 构建、测试与开发命令
- 环境安装：`pixi install -e dev` 然后 `pixi run -e dev lerobot`（lerobot 走单独 task，因为 pixi 不支持 `--no-deps`，详见 `pixi.toml`）
- 进入开发环境：`pixi shell -e dev`
- 类型检查：`pixi run typecheck`（pyright）
- 代码质量：`pre-commit install && pre-commit run -a`（ruff 格式化+lint，nbstripout 清理笔记本）
- 运行测试：`pytest -q`
- 示例运行：`python scripts/inference.py --config-name pi05_airbot --checkpoint-dir <ckpt>`

## 代码风格与命名约定
- Python 使用 4 空格缩进，`line-length=120`（ruff）。
- 导入顺序由 ruff/isort 管控，`first-party = {pi, common}`。
- 命名：模块/函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_CASE`。
- 注释与文档一律使用英文；仅为关键路径添加注释（例如性能敏感或复杂逻辑处）。

## 测试规范
- CPU 测试位于 `tests/`，运行 `pytest -q` 即可。覆盖 mask 构造、`norm_stats` 归一化、
  `_make_bool_mask`、`_apply_delta_actions`、数据 transform 拼装等纯函数。
- 涉及 GPU / CUDA / 分布式的测试不在本地跑：请给出明确的 `torchrun` 指令，由维护者在
  GPU 环境下执行。
- 新增 GPU 用例放在 `tests/operator/` 或 `tests/gpu/`，并在文件首行加注释说明所需硬件。

## 提交与合并请求
- 提交信息遵循 Conventional Commits：`type(scope): summary`。
  - 例：`feat(training): add Pi05 config`、`fix(models): handle dtype mismatch`。
- PR 要求：
  - 清晰描述与动机，关联 Issue（如 `#123`）。
  - 变更范围与影响面；必要的日志/截图/性能数据。
  - 本地验证通过：`pre-commit`, `pixi run typecheck`, `pytest -q`。

## 安全与配置提示
- 不要提交密钥与私有数据；使用环境变量与本地配置。
- 检查点与大型二进制放置于 `assets/` 或外部存储（GCS/LFS），遵守 `.gitignore`。
- 提交前确保笔记本已被 `nbstripout` 清理。

