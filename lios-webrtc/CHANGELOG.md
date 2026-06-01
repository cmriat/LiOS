# Changelog

All notable changes to this project will be documented in this file.

This file is maintained with [git-cliff](https://git-cliff.org); see `cliff.toml`.
Regenerate with: `git cliff --output CHANGELOG.md`

## Unreleased

### Engineering / 工程化

- 添加 Apache-2.0 LICENSE 与版权声明
- 补全 `pyproject.toml` 元数据（description / readme / license / authors / keywords / classifiers / urls）
- 添加 GitHub Actions CI：ruff lint + pytest + signal-server `go build`
- 添加 pytest 单元测试（`inference_buffer_v2` base64 往返与 POSIX 信号量、`gpu_sink` 纯逻辑），GPU 用例以 `-m gpu` 选跑
- 接入 ruff：修复 import 卫生，统一 lint 配置
- README 添加许可证段与引用（BibTeX）段
- 仓库清洁：归拢散落的 e2e 脚本到 `tests/e2e/`，清理失效的 `.gitignore` 规则
- 添加社区治理文件：`CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` / `SECURITY.md` / Issue + PR 模板

### Documentation

- 整理文档目录结构：合并 `docs/usages` → `docs/usage`，设计/用法文档双向交叉链接，`design.md` 升级为文档导航并优化架构图配色
- 品牌更新为 **LiOS**（README / `design.md` / 图表 / citation）；新增简体中文 README 与中英双向切换；顶部加 Logo（`logo/`）与徽章
- 突出低时延定位：README 引言强调 GPU 全链路直通与近网中继，延迟段前置、Quick start 前移

### Benchmark

- LiOS vs LiveKit 吞吐与延迟基准套件、跨机脚本与对比图表
