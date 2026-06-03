# 部署

[English](./deployment.md) | [中文](./deployment.zh-CN.md)

Pi 0.5 VLA 推理栈的云端部署指南。

## 适用范围

本文档定义 Pi VLA 推理栈的云端组件及其对外提供的服务接口。端侧实现不在本文档讨论
范围内；本文档仅规定端侧实现为达成互操作所必须提供或消费的接口。端侧栈的内部设计
——机器人控制中间件、电机驱动、运动学求解、安全策略、时间同步方式——由实现方自行决定。

> **无端侧栈的冒烟测试。** 若尚无端侧实现，可使用 `scripts/inference.py` 加载 checkpoint
> 并对一个 LeRobot 数据样本输出一条动作块：
>
> ```bash
> pixi run -e dev python scripts/inference.py \
>     --config-name pi05_airbot \
>     --checkpoint-dir /path/to/checkpoints/<exp>/step_10000 \
>     --repo-id /path/to/lerobot_dataset
> ```
>
> 本文余下内容讨论完整的云端推理栈。

---

## 1. 总览

云端部署包含三个长驻进程，运行于具备 GPU 的主机：

| 组件 | 默认端口 | 协议 | 职责 |
|---|---|---|---|
| 信令服务 | `18080` | WebSocket | 端侧 sender 与云端 receiver 之间的 WebRTC SDP/ICE 协商 |
| 图传 receiver | `5082` | HTTP | 解码来自 WebRTC 的视频流，将最新帧以序列化缓冲区形式对外暴露 |
| VLA 推理服务 | `18081` | WebSocket | 运行 VLA 模型；接收端侧关节状态，发布动作块（action chunk） |

数据流：

```
[端侧相机] ──WebRTC──> [云端 receiver] ──HTTP buffer──> [VLA 推理] ──WS action chunks──> [端侧客户端]
                                                              ▲
                                                              │ WS joint states
                                                              │
                                                        [端侧客户端]
```

---

## 2. 端侧需要实现的接口

云端部署通过三类服务接口与端侧交换数据。端侧实现需提供并消费下列接口；具体实现方
式由实现方决定。

### 2.1 媒体上行接口（端侧产出）

端侧实现需将 WebRTC sender 注册到云端信令服务（`ws://<云端>:18080`），并将 N 路相
机流推送给云端 receiver。流名需与 receiver 配置一致（默认：`mid`、`left`、`right`）。
当直连不可用时，ICE 协商需经过指定的 TURN 服务器。

### 2.2 状态上行接口（端侧产出）

端侧实现需对 `ws://<云端>:18081/ws/from-client` 维持一条长连接 WebSocket，按
50–100 Hz 频率发布符合 §3.2 的关节状态消息。连接断开后应实现指数退避重连。

### 2.3 动作下行接口（端侧消费）

端侧实现需对 `ws://<云端>:18081/ws/to-client` 维持一条长连接 WebSocket，消费符合
§3.3 的动作块消息。端侧控制器负责动作块之间的时间对齐、插值，以及安全约束（关节
限位、软停、运动学校验）。云端不施加任何安全策略。

---

## 3. 通信协议

### 3.1 HTTP 图像缓冲

图传 receiver 暴露以下端点：

```
GET /api/v1/healthz                → 200  application/json   {"status":"ok"}
GET /api/v1/infer-buffer/base64    → 200  text/plain         <base64 编码的负载>
```

base64 负载解码后为序列化的 `InferenceBufferV2` 实例，包含每路已注册相机的最新帧。
帧为形状 `(H, W, 3)` 的 RGB `uint8` 数组，由 receiver 缩放至 `224×224`。

### 3.2 关节状态消息（端侧 → 云端）

**端点：** `ws://<云端>:18081/ws/from-client`
**帧格式：** UTF-8 JSON 文本帧。

```json
{
  "timestamp/ms": 1780309102254.69,
  "left_arm":  { "positions": [j0, j1, j2, j3, j4, j5, gripper],
                 "velocities": [j0, j1, j2, j3, j4, j5, gripper] },
  "right_arm": { "positions": [j0, j1, j2, j3, j4, j5, gripper],
                 "velocities": [j0, j1, j2, j3, j4, j5, gripper] }
}
```

| 字段 | 类型 | 单位 | 是否必填 | 备注 |
|---|---|---|---|---|
| `timestamp/ms` | float | 毫秒（自 epoch） | 是 | 状态采样时的墙钟时间 |
| `*.positions` | 7 个 float | 弧度（关节）；归一化开度（夹爪） | 是 | 第 7 位（索引 6）为夹爪 |
| `*.velocities` | 7 个 float | 弧度/秒 | 否 | 建议提供，便于诊断 |

推理服务采用深度为 1 的 leaky 队列，丢弃被覆盖的旧消息。超频发布不会造成损害。

### 3.3 动作块消息（云端 → 端侧）

**端点：** `ws://<云端>:18081/ws/to-client`
**帧格式：** UTF-8 JSON 文本帧。

```json
{
  "control_mode": "joint",
  "timestamp/ms": 1780309102254.69,
  "delta/ms": 50,
  "left_arm":  { "actions": [[a0, a1, a2, a3, a4, a5, gripper], ... × HORIZON] },
  "right_arm": { "actions": [[a0, a1, a2, a3, a4, a5, gripper], ... × HORIZON] },
  "send_time": 1780309102.25
}
```

| 字段 | 类型 | 单位 | 备注 |
|---|---|---|---|
| `control_mode` | string | — | 当前固定为 `"joint"`。端侧控制器需校验此值与其配置的输入类型一致 |
| `timestamp/ms` | float | 毫秒 | 块的参考时刻；第 i 步目标时刻为 `timestamp + i * delta` |
| `delta/ms` | float | 毫秒 | 块内相邻步之间的时间间隔 |
| `*.actions` | `[HORIZON, 7]` float | 弧度（关节）；已反归一化、已 delta→absolute 转换后的绝对值 | `HORIZON` 默认 30 |
| `send_time` | float | 秒（自 epoch） | 云端发送时刻；用于测量下行延迟 |

发布频率约为 `1 / inference_latency`，在单卡 A100/H100 级别 GPU 上典型为 8–12 Hz。

---

## 4. 配置

部署相关的配置位于 `scripts/deployment/inference.ipynb` 顶部几个 cell。以下片段反
映 notebook 当前的实际写法：

```python
HORIZON = 30
NUM_STEPS = 40
DISCRETE_STATE_INPUT = True

config = TrainConfig(
    name="pi05_airbot",
    model=pi_config.PiConfig(
        pi05=True,
        action_horizon=HORIZON,
        discrete_state_input=DISCRETE_STATE_INPUT,
    ),
    data=DatasetConfig(
        repo_id="path_to_lerobot_dataset",
        asset_id="airbot",
        apply_delta_transform=True,
        use_quantile_norm=True,
        action_sequence_keys=("action",),
        state_sequence_keys=("observation.state",),
    ),
    batch_size=1,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=5_000, peak_lr=5e-5, decay_steps=500_000, decay_lr=5e-5,
    ),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    ema_decay=0.999,
    pytorch_weight_path=None,
    overwrite=True,  # 必须：TrainConfig.__post_init__ 要求 overwrite=True 或 resume=True，否则直接抛错
)

ckpt_path      = "path_to_checkpoint"
norm_stat_path = "path_to_norm_stats_dir"
```

| 占位符 | 应替换为 | 说明 |
|---|---|---|
| `path_to_lerobot_dataset` | LeRobot 数据集目录的文件系统路径 | **运行时会真正加载。** notebook 通过 `pi.data.SimpleLeRobotLoader(repo_id=...)` 构造，内部会实例化 `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` 取一条样本用于 prompt 文本、image-mask 布局和 tokenizer 初始化。路径不存在或为空会在 dataloader 构造阶段直接报错。 |
| `path_to_checkpoint` | 微调后 checkpoint 的目录路径（`torch.distributed.checkpoint` 格式） | 通过 `torch.distributed.checkpoint.load(model.state_dict(), checkpoint_id=ckpt_path, planner=MetadataNormalizingPlanner())` 加载。详见 §4.1。 |
| `path_to_norm_stats_dir` | 包含 `norm_stats.json` 的目录 | 直接通过 `pi.shared.normalize.load(norm_stat_path)` 读取。notebook **不走** `config.data.load_norm_stats(config.assets_dirs)`，而是用这条绝对路径独立加载，与 config 无关。 |

### 4.1 `MetadataNormalizingPlanner`

`scripts/deployment/normalizer.py` 定义了一个 `DefaultLoadPlanner` 的子类，作用是
在 checkpoint metadata 的 key 匹配 live `state_dict` 之前剥掉以下前缀：

```
_orig_mod.        ._orig_mod
._fsdp_wrapped_module
._checkpoint_wrapped_module
.module           _module.
```

只要 checkpoint 是在训练时被 `torch.compile`、FSDP 或 activation checkpointing 包
裹过的模型保存出来的，加载时就必须带这个 planner。否则裸
`torch.distributed.checkpoint.load(...)` 会因 key 不匹配而报错。

---

## 5. 启动

三个进程需按以下顺序启动，每个进程独占一个会话：

1. **信令服务。** 端侧 sender 注册前必须已在监听。
2. **图传 receiver。** 推理服务启动前必须已在运行并接收到帧。
3. **VLA 推理服务。** 推理入口是 `scripts/deployment/inference.ipynb` 这个 notebook。
   从上到下依次执行其 cell 即可：加载模型、构造 dataloader、启动动作 WebSocket
   服务、拉取首帧图像 buffer、执行 warmup（第一轮 warmup 触发 `torch.compile`，
   可能耗时 30 秒至 2 分钟），然后进入 `while True: get_action(data)` 主循环。

notebook 通过 `JSONWebSocketAPIServer(host="0.0.0.0", port=18081)` 构造动作 WS 服
务。当端侧客户端位于其他主机时，该地址需通过直连、VPN 或端口转发可达（参见 §7）。

### 5.1 主机级 HTTP 代理

若云端主机存在进程级 HTTP 代理（如 Clash、V2Ray），`websockets` 库可能将
WebSocket 流量路由到代理，导致握手失败。启动推理服务前：

```sh
export NO_PROXY="127.0.0.1,localhost,::1"
# 或对当前 shell：
unset http_proxy https_proxy
```

---

## 6. 无端侧实现的测试

最小化 mock 客户端可在无端侧实现的情况下完成云端部署的端到端验证。Mock 客户端以
恒定关节状态和被动动作消费者的形式实现 §3.2 与 §3.3 的接口：

```python
# mock_edge_client.py
import asyncio, json, time
import websockets

WS = "ws://127.0.0.1:18081"

async def send_states():
    async with websockets.connect(f"{WS}/ws/from-client") as ws:
        while True:
            await ws.send(json.dumps({
                "timestamp/ms": time.time() * 1000,
                "left_arm":  {"positions": [0.0] * 7, "velocities": [0.0] * 7},
                "right_arm": {"positions": [0.0] * 7, "velocities": [0.0] * 7},
            }))
            await asyncio.sleep(0.01)

async def recv_actions():
    async with websockets.connect(f"{WS}/ws/to-client") as ws:
        async for msg in ws:
            data = json.loads(msg)
            n = len(data.get("left_arm", {}).get("actions", []))
            print(f"[action] ts={data['timestamp/ms']} steps={n} mode={data['control_mode']}")

asyncio.run(asyncio.gather(send_states(), recv_actions()))
```

仍需图像来源。若无可用的 WebRTC sender，可用返回合成 `InferenceBufferV2` 负载的
mock HTTP 服务替代 receiver。

Mock 测试可验证模型加载、warmup 行为、推理延迟、WebSocket 协议合规性、JSON 格式、
以及输出动作块的数值合理性。Mock 测试**不能**验证在真实或仿真机器人上的端到端
执行行为。

若无需启动任何云端进程，可使用 `scripts/inference.py` 进行纯离线冒烟测试（参见文
档顶部的提示框）。

---

## 7. 网络隧道

当云端主机与端侧不同机时，端侧可直接连接云端公网地址（建议在私网或 VPN 内使用），
或建立 SSH 隧道：

```sh
ssh -N \
    -L 18080:127.0.0.1:18080 \
    -L 18081:127.0.0.1:18081 \
    -L 5082:127.0.0.1:5082  \
    <云端主机>
```

| 转发端口 | 由谁使用 | 用途 |
|---|---|---|
| `18080` | 端侧 WebRTC sender | 发现云端信令服务 |
| `18081` | 端侧 WebSocket 客户端 | 动作通道连通性 |
| `5082` | 操作员（可选） | 通过浏览器查看接收到的解码帧 |

---

## 8. 排障

云端侧可观察到的现象及典型原因：

| 现象 | 原因 | 解决方法 |
|---|---|---|
| `FileNotFoundError: Norm stats file not found at ...norm_stats.json` | `norm_stat_path` 没有指向包含 `norm_stats.json` 的目录 | 直接核对 `norm_stat_path`；notebook 在这一步不走 `config.assets_dirs` |
| `TrainConfig(...)` 时报 `Must set either --overwrite ... or --resume ...` | 漏了 `overwrite=True`（或 `resume=True`） | 加上 `overwrite=True`。notebook 不写 checkpoint，这只是个安全开关 |
| 加载 ckpt 的 cell 抛 `ModuleNotFoundError: No module named 'normalizer'` | `scripts/deployment/normalizer.py` 缺失，或 notebook 的 `sys.path.append(...)` 没先跑到 | 从 git 恢复 `normalizer.py`，并确保 notebook 里那行 `sys.path.append` 已经执行 |
| 展示 `buffer.images` 时抛 `KeyError: 'right'`（或 `'left'` / `'mid'`） | `infer-buffer/base64` 返回了一个没有任何相机帧的 buffer（HEALTHZ 200 但 `len ≈ 200`），或 receiver 注册的 stream 名字不一样 | 5082 receiver 在线但没有 edge 端 WebRTC sender 在推流 —— 检查 sender 进程、信令@18080、以及 `--streams N` 配置 |
| lerobot 的 `decode_video_frames` 抛 `AttributeError: module 'torchvision.io' has no attribute 'VideoReader'` | 当前 pixi 环境里没装 `torchcodec`；lerobot 的 pyav 兜底分支同样依赖 `torchvision.io.VideoReader`，而当前 torchvision build 没编 video 支持 | 将 `torchcodec` 安装至 notebook kernel 使用的环境 |
| `HEALTHZ 200` 后 `GET_BASE64` 超时 | receiver 已启动但无 WebRTC sender 接入 | 检查端侧 sender 是否成功注册到信令服务 |
| 推理延迟高于 200 ms | GPU 资源不足，或 `torch.compile` 降级为 eager 模式 | 检查 GPU 利用率及 TorchDynamo 警告 |
| WebSocket 握手超时 | 主机级 HTTP 代理拦截了 loopback 连接 | 将 `NO_PROXY` 设置为包含 localhost |
| 外部客户端无法连接 `:18081` | 服务仅绑定到 `127.0.0.1` | 构造 `JSONWebSocketAPIServer` 时配置 `host="0.0.0.0"` |
| 动作块已发布但端侧未响应 | `control_mode` 与端侧控制器配置不匹配 | 调整端侧控制器输入类型与动作块 `control_mode` 一致 |

---

## 9. 协议版本

当前协议版本规定以下不变量：

| 参数 | 取值 |
|---|---|
| `control_mode` | `"joint"` |
| `HORIZON` | 30 |
| `delta/ms` | 50 |
| 状态维度 | 单臂 7 维（6 关节 + 1 夹爪），双臂 |
| 动作维度 | 单臂 7 维（6 关节 + 1 夹爪），双臂 |

对上述任一参数的修改均构成协议变更，端云两侧需协同更新。修改 `HORIZON` 或
`delta/ms` 将使端侧控制器的时间对齐假设失效，可能导致已发布的动作块被判定为过期。

---

## 附录 A — 关键源文件

| 路径 | 用途 |
|---|---|
| `scripts/deployment/inference.ipynb` | 主推理 notebook（本目录下唯一的 `.ipynb`） |
| `scripts/deployment/normalizer.py` | `MetadataNormalizingPlanner` —— 针对 compile/FSDP/AC 包裹模型的 checkpoint key 标准化 |
| `src/pi/services/ws_json_api.py` | WebSocket 服务实现（`JSONWebSocketAPIServer`，在绑定端口上同时提供 `/ws/from-client` 与 `/ws/to-client`） |
| `src/pi/inference_buffer_v2.py` | `InferenceBufferV2` 数据结构 |
| `src/pi/shared/normalize.py` | `pi.shared.normalize.load()` —— 从目录读取 `norm_stats.json` |

## 附录 B — 默认端口

| 端口 | 协议 | 服务 | 默认绑定 |
|---|---|---|---|
| 18080 | WebSocket | 信令服务 | `0.0.0.0` |
| 5082 | HTTP | 图传 receiver | `0.0.0.0` |
| 18081 | WebSocket | VLA 推理 | notebook 绑定到 `0.0.0.0`，方便异机端侧客户端连接 |

## 相关文档

- [架构](./architecture.zh-CN.md) —— 推理服务实际运行的内容
- [训练](./training.zh-CN.md) —— 推理服务加载的 checkpoint 的产生流程
