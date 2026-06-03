# Deployment

[English](./deployment.md) | [中文](./deployment.zh-CN.md)

Cloud-side deployment guide for the Pi 0.5 VLA inference stack.

## Scope

This document specifies the cloud-side components of the Pi VLA inference stack and
the service interfaces they expose. Edge-side implementations are out of scope; this
document only states the interfaces an edge implementation must produce or consume in
order to interoperate. The internal design of the edge stack — robot control
middleware, motor drivers, kinematics, safety policy, time-synchronisation strategy —
is implementation-defined.

> **Headless smoke test.** When no edge stack is available yet, `scripts/inference.py`
> loads a checkpoint and prints one action chunk from a LeRobot dataset sample:
>
> ```bash
> pixi run -e dev python scripts/inference.py \
>     --config-name pi05_airbot \
>     --checkpoint-dir /path/to/checkpoints/<exp>/step_10000 \
>     --repo-id /path/to/lerobot_dataset
> ```
>
> The rest of this document covers the live cloud deployment stack.

---

## 1. Overview

The cloud deployment consists of three long-lived processes running on a GPU-enabled host:

| Component | Default port | Protocol | Role |
|---|---|---|---|
| Signaling server | `18080` | WebSocket | WebRTC SDP/ICE negotiation between the edge sender and the cloud receiver |
| Image receiver | `5082` | HTTP | Decodes incoming WebRTC video and exposes the latest frames as a serialised buffer |
| VLA inference service | `18081` | WebSocket | Runs the VLA model; receives joint states from the edge and publishes action chunks |

Data flow:

```
[edge cameras] ──WebRTC──> [cloud receiver] ──HTTP buffer──> [VLA inference] ──WS action chunks──> [edge client]
                                                                  ▲
                                                                  │ WS joint states
                                                                  │
                                                              [edge client]
```

---

## 2. Edge-side Interface Requirements

The cloud deployment exchanges data with the edge through three service interfaces.
An edge implementation is required to produce and consume these interfaces; the
internal realisation is left to the implementer.

### 2.1 Media-uplink interface (edge produces)

The edge implementation registers a WebRTC sender with the cloud signaling server
(`ws://<cloud>:18080`) and publishes N camera streams to the cloud receiver. Stream
names must match the receiver configuration (default: `mid`, `left`, `right`). ICE
negotiation must use the configured TURN server when direct connectivity is
unavailable.

### 2.2 State-uplink interface (edge produces)

The edge implementation maintains a persistent WebSocket connection to
`ws://<cloud>:18081/ws/from-client` and publishes joint state messages conforming to
§3.2 at 50–100 Hz. The connection should implement reconnect-with-backoff on
disconnection.

### 2.3 Action-downlink interface (edge consumes)

The edge implementation maintains a persistent WebSocket connection to
`ws://<cloud>:18081/ws/to-client` and consumes action chunk messages conforming to
§3.3. The edge controller is responsible for time alignment, interpolation between
chunks, and safety enforcement (joint limits, soft stops, kinematic validation). The
cloud applies no safety policy.

---

## 3. Wire Protocols

### 3.1 HTTP Camera Buffer

The image receiver exposes the following endpoints:

```
GET /api/v1/healthz                → 200  application/json   {"status":"ok"}
GET /api/v1/infer-buffer/base64    → 200  text/plain         <base64-encoded payload>
```

The base64 payload decodes to a serialised `InferenceBufferV2` instance containing
the latest frame from each registered camera. Frames are RGB `uint8` arrays of shape
`(H, W, 3)`, resized to `224×224` by the receiver.

### 3.2 Joint State Message (Edge → Cloud)

**Endpoint:** `ws://<cloud>:18081/ws/from-client`
**Frame format:** UTF-8 JSON text frames.

```json
{
  "timestamp/ms": 1780309102254.69,
  "left_arm":  { "positions": [j0, j1, j2, j3, j4, j5, gripper],
                 "velocities": [j0, j1, j2, j3, j4, j5, gripper] },
  "right_arm": { "positions": [j0, j1, j2, j3, j4, j5, gripper],
                 "velocities": [j0, j1, j2, j3, j4, j5, gripper] }
}
```

| Field | Type | Unit | Required | Notes |
|---|---|---|---|---|
| `timestamp/ms` | float | milliseconds since epoch | Yes | Wall-clock at state acquisition |
| `*.positions` | array of 7 floats | radians (joints), normalised opening (gripper) | Yes | Index 6 is the gripper |
| `*.velocities` | array of 7 floats | radians/second | No | Recommended for diagnostics |

The inference service uses a leaky queue of depth 1 and discards superseded messages.
Over-publication is harmless.

### 3.3 Action Chunk Message (Cloud → Edge)

**Endpoint:** `ws://<cloud>:18081/ws/to-client`
**Frame format:** UTF-8 JSON text frames.

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

| Field | Type | Unit | Notes |
|---|---|---|---|
| `control_mode` | string | — | Currently `"joint"`. The edge controller must validate that this matches its configured input type. |
| `timestamp/ms` | float | milliseconds | Reference epoch for the chunk; the i-th step targets `timestamp + i * delta` |
| `delta/ms` | float | milliseconds | Inter-step interval within the chunk |
| `*.actions` | `[HORIZON, 7]` floats | radians (joints), absolute, post-denormalisation | `HORIZON` defaults to 30 |
| `send_time` | float | seconds since epoch | Cloud-side send timestamp; useful for downlink-latency measurement |

Publication rate is approximately `1 / inference_latency`, typically 8–12 Hz on a
single A100/H100-class GPU.

---

## 4. Configuration

The deployment-specific values live in the early cells of
`scripts/deployment/inference.ipynb`. The snippet below reflects the notebook as
currently coded:

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
    overwrite=True,  # required: TrainConfig.__post_init__ refuses to instantiate without overwrite=True or resume=True
)

ckpt_path      = "path_to_checkpoint"
norm_stat_path = "path_to_norm_stats_dir"
```

| Placeholder | Replacement | Notes |
|---|---|---|
| `path_to_lerobot_dataset` | Filesystem path to a LeRobot dataset directory | **Actually loaded at runtime.** The notebook constructs `pi.data.SimpleLeRobotLoader(repo_id=...)`, which in turn instantiates `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` to fetch one sample for prompt text, image-mask layout, and tokenizer setup. A missing or empty path fails at dataloader construction. |
| `path_to_checkpoint` | Filesystem path to the fine-tuned checkpoint directory in `torch.distributed.checkpoint` format | Loaded via `torch.distributed.checkpoint.load(model.state_dict(), checkpoint_id=ckpt_path, planner=MetadataNormalizingPlanner())`. See §4.1. |
| `path_to_norm_stats_dir` | Directory containing `norm_stats.json` | Loaded directly via `pi.shared.normalize.load(norm_stat_path)`. The notebook does **not** go through `config.data.load_norm_stats(config.assets_dirs)`; it reads from this absolute path independent of the config. |

### 4.1 `MetadataNormalizingPlanner`

`scripts/deployment/normalizer.py` defines a `DefaultLoadPlanner` subclass that
strips the following prefixes from checkpoint metadata keys before they are matched
against the live `state_dict`:

```
_orig_mod.        ._orig_mod
._fsdp_wrapped_module
._checkpoint_wrapped_module
.module           _module.
```

This is required whenever the checkpoint was produced by a model that was wrapped
with `torch.compile`, FSDP, or activation checkpointing during training. Loading such
a checkpoint without the planner raises mismatched-key errors against the unwrapped
inference model.

---

## 5. Startup

The three processes must be started in the following order, each in a dedicated
session:

1. **Signaling server.** Must be listening before the edge sender attempts to
   register.
2. **Image receiver.** Must be running and receiving frames before the inference
   service is started.
3. **VLA inference service.** The notebook `scripts/deployment/inference.ipynb` is
   the inference entry point. Running its cells top-to-bottom loads the model,
   constructs the dataloader, starts the action WebSocket server, fetches an initial
   image buffer, runs warmup (the first warmup iteration triggers `torch.compile`,
   which can take 30 s–2 min), and then enters the `while True: get_action(data)`
   loop.

The notebook constructs `JSONWebSocketAPIServer(host="0.0.0.0", port=18081)`. When
the edge client resides on a different host, this address must be reachable via
direct connectivity, VPN, or port forwarding (see §7).

### 5.1 Host-level HTTP Proxy

If the cloud host has a process-level HTTP proxy configured (e.g. Clash, V2Ray), the
`websockets` library may route WebSocket traffic through the proxy and fail the
handshake. Before starting the inference service:

```sh
export NO_PROXY="127.0.0.1,localhost,::1"
# or, for the current shell:
unset http_proxy https_proxy
```

---

## 6. Testing Without an Edge Implementation

A minimal mock client allows end-to-end validation of the cloud deployment without
requiring an edge implementation. The mock conforms to the interfaces of §3.2 and
§3.3 with a stationary joint state and a passive action consumer:

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

A camera source remains required. When no WebRTC sender is available, the receiver
may be replaced with a mock HTTP server returning a synthetic `InferenceBufferV2`
payload.

Mock-based testing validates the model load, warmup behaviour, inference latency,
WebSocket protocol compliance, JSON formatting, and numerical sanity of the published
action chunks. It does not validate end-to-end execution on a physical or simulated
robot.

For a fully headless smoke test that does not require any of the three cloud
services to be running, use `scripts/inference.py` (see the callout at the top of
this document).

---

## 7. Network Tunnelling

When the cloud host is remote from the edge, the edge may either connect directly to
the cloud's public addresses (recommended within a private network or VPN) or
establish an SSH tunnel:

```sh
ssh -N \
    -L 18080:127.0.0.1:18080 \
    -L 18081:127.0.0.1:18081 \
    -L 5082:127.0.0.1:5082  \
    <cloud-host>
```

| Forwarded port | Required by | Purpose |
|---|---|---|
| `18080` | Edge WebRTC sender | Discovering the cloud signaling server |
| `18081` | Edge WebSocket client | Action-channel connectivity |
| `5082` | Operator (optional) | Inspecting decoded frames from a browser |

---

## 8. Troubleshooting

Symptoms observable from the cloud side and their typical causes:

| Symptom | Cause | Resolution |
|---|---|---|
| `FileNotFoundError: Norm stats file not found at ...norm_stats.json` | `norm_stat_path` does not point to a directory containing `norm_stats.json` | Verify `norm_stat_path` directly; the notebook does not use `config.assets_dirs` here |
| `Must set either --overwrite (to start new training) or --resume ...` at `TrainConfig(...)` | `overwrite=True` (or `resume=True`) was omitted | Set `overwrite=True` (the notebook never writes checkpoints, so this is purely a safety toggle) |
| `ModuleNotFoundError: No module named 'normalizer'` at the ckpt-load cell | `scripts/deployment/normalizer.py` missing, or `sys.path` does not include `scripts/deployment` | Restore `normalizer.py` from git and ensure the notebook's `sys.path.append(...)` runs first |
| `KeyError: 'right'` (or `'left'`, `'mid'`) when displaying `buffer.images` | `infer-buffer/base64` returned a buffer with no camera frames (HEALTHZ 200 but `len ≈ 200`), or the receiver registered different stream names | The 5082 receiver is online but no edge WebRTC sender is publishing — check sender process + signaling@18080 + `--streams N` |
| `RuntimeError: torchvision.io has no attribute 'VideoReader'` from lerobot's `decode_video_frames` | `torchcodec` not installed in the active pixi environment; lerobot's pyav fallback also reaches into `torchvision.io.VideoReader`, which is absent in the current build | Install `torchcodec` into the env that the notebook kernel uses |
| `HEALTHZ 200` followed by `GET_BASE64` timeouts | Receiver running but no WebRTC sender connected | Verify edge sender registration with the signaling server |
| Inference latency above 200 ms | GPU under-provisioned, or `torch.compile` degraded to eager mode | Inspect GPU utilisation and TorchDynamo warnings |
| WebSocket handshake timeout | Host-level HTTP proxy intercepting loopback connections | Set `NO_PROXY` to include localhost |
| External clients fail to connect to `:18081` | Service bound to `127.0.0.1` only | Configure `host="0.0.0.0"` when constructing `JSONWebSocketAPIServer` |
| Action chunks published but ignored by the edge | `control_mode` mismatch with the edge controller configuration | Align the edge controller's input type with the chunk's `control_mode` |

---

## 9. Protocol Versioning

The current protocol revision specifies the following invariants:

| Parameter | Value |
|---|---|
| `control_mode` | `"joint"` |
| `HORIZON` | 30 |
| `delta/ms` | 50 |
| State dimensions | 7 per arm (6 joints + 1 gripper), 2 arms |
| Action dimensions | 7 per arm (6 joints + 1 gripper), 2 arms |

Any deviation from these parameters constitutes a protocol change and requires
coordinated updates on both sides. Changes to `HORIZON` or `delta/ms` will invalidate
time-alignment assumptions on the edge controller and may cause published action
chunks to be treated as expired.

---

## Appendix A — Key Source Files

| Path | Purpose |
|---|---|
| `scripts/deployment/inference.ipynb` | Main inference notebook (only `.ipynb` in this directory) |
| `scripts/deployment/normalizer.py` | `MetadataNormalizingPlanner` — checkpoint-key normalisation for compiled/FSDP/AC-wrapped models |
| `src/pi/services/ws_json_api.py` | WebSocket server implementation (`JSONWebSocketAPIServer`, exposes `/ws/from-client` and `/ws/to-client` on the bound port) |
| `src/pi/inference_buffer_v2.py` | `InferenceBufferV2` data structure |
| `src/pi/shared/normalize.py` | `pi.shared.normalize.load()` — reads `norm_stats.json` from a directory |

## Appendix B — Default Ports

| Port | Protocol | Service | Default bind |
|---|---|---|---|
| 18080 | WebSocket | Signaling server | `0.0.0.0` |
| 5082 | HTTP | Image receiver | `0.0.0.0` |
| 18081 | WebSocket | VLA inference | Bound to `0.0.0.0` by the notebook so off-host edge clients can connect |

## See also

- [Architecture](./architecture.md) — what the inference service actually runs
- [Training](./training.md) — producing the checkpoint that the inference service loads
