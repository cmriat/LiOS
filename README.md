# LiOS

LiOS is a stack for embodied-AI infrastructure. This repository holds its
subprojects side by side.

## Subprojects

| Path | What it is |
|---|---|
| [`lios-pi/`](lios-pi/) | A pure-PyTorch implementation of the **Pi0** and **Pi0.5** vision-language-action (VLA) models, ported from [openpi](https://github.com/Physical-Intelligence/openpi): first-class FSDP training, LeRobot dataset support, and a reference WebRTC + WebSocket inference stack. See [`lios-pi/README.md`](lios-pi/README.md) and the topic docs under [`lios-pi/docs/`](lios-pi/docs/). |
| [`lios-webrtc/`](lios-webrtc/) | Low-latency, GPU-accelerated edge → cloud **image transport** for robot inference: WebRTC + GStreamer sender / receiver, Go signaling server, CUDA-IPC zero-copy inference buffer, MJPEG HTTP live preview, benchmarks. See [`lios-webrtc/README.md`](lios-webrtc/README.md) and the [user guide](lios-webrtc/USER_GUIDE.md). |

More subprojects (e.g. `data/`) will land alongside as the stack grows.

## License

[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the `LICENSE` (and `NOTICE`, where present) file in each subproject.
