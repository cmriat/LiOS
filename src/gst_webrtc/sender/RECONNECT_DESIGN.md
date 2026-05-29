# Sender 断连快速恢复设计

> 通用 webrtcbin 恢复策略（分层恢复 + O/A + ICE restart 时序）见 [`docs/design/RESTORE_DESIGN.md`](../../../docs/design/RESTORE_DESIGN.md)。

问题现象（用户复现）
- Receiver 退出后，Sender 仍保持运行；等待一段时间再启动新的 Receiver，双方无法完成握手与推流。
- Sender 侧日志间歇出现：
  - `rtpsource: running time not set, can not create SR ...`
  - `rtpsession: generated empty RTCP messages for all the sources`
  - `[webrtc] ICE restart: creating offer with ice-restart=true` 之后偶尔 `connection state: connected`，但链路依旧没有恢复。

根因分析
- 仅做 ICE Restart 不足以覆盖「对端真正离开」的场景；更关键的是，Sender 没有在 `peer-leave` 后清理当前的 `_remote_id`，也没有在新对端加入后重新发起一次**干净的** O/A。
- `rtpsource/rtpsession` 的告警大多是 webrtcbin 内部 RTP 定时器在链路断开时无法生成有效 RTCP，属于症状而非根因；彻底重建 webrtcbin（或 reset 到 READY/NULL）即可消除。

目标
- 快速感知对端离线（不要“等很久”）并做出反应。
- Receiver 重新运行后，能够顺利与 Sender 重新握手并开始推流。

设计概览（L0→L3 分层恢复）
1) L0 观测：
   - 监听 webrtcbin 的 `notify::connection-state` 与 `notify::ice-connection-state`。
   - 监听信令 `peer-leave` / `peers` 列表变化，判定对端是否真的离开房间。

2) L1 快速重试（ICE Restart）：
   - 当 `ice-connection-state` 进入 `disconnected` 持续 > 1.0s 或进入 `failed`，发起 `create-offer`（携带 `ice-restart=true`）并走一轮 O/A。
   - 设置 5s 截止时间，如果 5s 内未回到 `connected`，升级到 L3 完全重连。

3) L3 完全重连（Full Reset）：
   - 在 `peer-leave`（或 peers 列表中不再包含当前对端）时，立即触发；或 L1 超时后触发。
   - 操作：销毁旧 webrtcbin，重新创建 webrtcbin，保持所有 RTP 源（编码与打包链）不变，将其 `src` 重新链接到新 webrtcbin 的 `sink_%u`；随后等待新对端加入并重新发起 O/A。
   - 若 GStreamer ≥ 1.28，尝试对 webrtcbin 触发 `close` action；否则将旧 webrtcbin 切到 `READY` 并从 pipeline 移除。

关键实现点
- 信令侧：
  - 处理 `peer-leave`：若离开的正是当前 `_remote_id`，清空 `_remote_id` 并执行 `_full_reset_webrtc()`。
  - 处理 `peers`：当前 `_remote_id` 不在列表中时等同于离开。
  - 处理 `peer-join`：当没有 `_remote_id` 时采用新的对端并立刻发起 O/A。

- WebRTC 侧：
  - `disconnected > 1.0s` 或 `failed` → 触发 `_restart_ice()`；并设置 5s 升级定时。
  - 若 5s 后仍未回到 `connected` → `_full_reset_webrtc()`。
  - `connected` 时清除所有定时与状态标记。

- Pipeline 侧：
  - 完全重连时仅替换 webrtcbin，复用、重连所有源到新的 webrtcbin；保持 pipeline 处于 `PLAYING`，通过 `sync_state_with_parent()` 让新 webrtcbin 追随状态。

落地改动
- 代码：`gst_webrtc/sender/core.py`
  - 新增：`peer-leave`/`peers` 处理，`_on_remote_left()`、`_full_reset_webrtc()`、升级定时 `_ice_restart_deadline` 与 `tick()`。
  - 将 ICE 断链阈值从 3s 收紧为 1s。
- 入口脚本：`gst_webrtc/sender.py`
  - 改为调用 `WebRTCSender`，并以与原先相同的相机→NVENC→H264→RTP 描述串添加视频源，统一走带恢复能力的实现。

运行步骤
- 启动信令：
  - `cd signal-server && go build -o signalsrv . && ./signalsrv serve --addr :18080`
- 启动 Sender：
  - `ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws python gst_webrtc/sender.py`
- 启动 Receiver（任一版本）：
  - 示例：`ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws python gst_webrtc/receiver.py`

验证要点
- Receiver 关闭时，Sender 立即打印 `peer left` 并在 100ms 内完成 webrtcbin 重建（日志：`webrtcbin rebuilt ...`）。
- Receiver 重新启动后，Sender 立刻向新对端发起 offer，完成 O/A 并进入 `connection state: connected`，视频恢复。
- 若仅网络抖动（`disconnected` 非真实离线），先尝试 ICE Restart（≤1s 触发），5s 内未恢复才执行 Full Reset。

已知限制与后续优化
- GI 版本低时 webrtcbin `close` action 可能不可用，代码已做保护分支，不影响功能。
- 如需多路对端选择/切换，可扩展 `_remote_id` 选择策略（当前取第一个可用）。
- 可将阈值（1s/5s）做成环境变量以便按现场网络调整。

