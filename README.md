# LiOS

LiOS is a stack for embodied-AI infrastructure. This repository holds its
subprojects side by side.

## Subprojects

| Path | What it is |
|---|---|
| [`lios-webrtc/`](lios-webrtc/) | Low-latency, GPU-accelerated edge → cloud **image transport** for robot inference: WebRTC + GStreamer sender / receiver, Go signaling server, CUDA-IPC zero-copy inference buffer, MJPEG HTTP live preview, benchmarks. See [`lios-webrtc/README.md`](lios-webrtc/README.md) and the [user guide](lios-webrtc/USER_GUIDE.md). |

More subprojects (e.g. `data/`) will land alongside as the stack grows.

## License

[Apache License 2.0](lios-webrtc/LICENSE)
