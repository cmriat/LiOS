# LiOS User Guide

Take the camera feed from a robot in the field, encode it with the GPU (or CPU) → transport it over WebRTC → decode it in the cloud, drop it into an
inference buffer to feed your model, and optionally watch the live feed in a browser. This guide covers it all: **how to run end-to-end, use a real camera,
view the live feed over HTTP, and every configuration option.**

> Convention: this repo's environment is managed by [pixi](https://pixi.sh), so all Python commands are prefixed with `pixi run`.

---

## 1. How it works (understand it in 30 seconds)

Three roles plus one external relay:

```
   [边缘机:sender]                                  [云端机:receiver]
   摄像头 → 编码(NVENC/x264) ──┐               ┌── 解码(NVDEC/avdec)→ 推理缓冲 → 模型
                               │               │                         └→ HTTP 实时预览
                               └─► signal-server(信令握手)◄─┘
              \──── 媒体流经 TURN/coturn 中继(SRTP) ────/
```

- **signal-server** (Go): just a "handshake broker" that forwards SDP/ICE. Both ends must be able to reach it.
- **sender / receiver** (Python): the two ends that actually send and receive the video.
- **TURN/coturn**: the relay that forwards the media stream (deploying your own near-network relay is the core feature of LiOS).

---

## 2. Set up the environment

```bash
pixi install          # Install all dependencies (GStreamer 1.26 / CUDA / PyTorch / Go ...)
```

External prerequisites:
- A machine that can run **signal-server** (both ends must be able to reach its IP:port).
- A **TURN/coturn** server (practically mandatory across machines/networks; you can skip it for same-machine localhost testing).
- **GPU optional**: with an NVIDIA GPU + the GStreamer nvcodec plugins (`nvh264enc`/`nvh264dec`/`cudaupload`)
  you get hardware encode/decode; without one it still works on CPU (`x264enc`/`avdec_h264`), see [section 7](#7-没有-gpu纯-cpu-跑).

---

## 3. Configuration: every `.env` parameter

Copy the template to `.env` (it's git-ignored, so you can put real secrets in it). Both examples load it automatically at startup
(`gst_webrtc.load_env()`; **existing environment variables take precedence over `.env`**):

```bash
cp .env.example .env
```

| Variable | Purpose | Default |
|---|---|---|
| `ROOM` | Room name, **must match on both ends**; this is how they find each other | `demo` |
| `SIGNAL_URL` | Signal server WebSocket address | `ws://127.0.0.1:18080/ws` |
| `STUN` | STUN server | `stun://stun.l.google.com:19302` |
| `TURN` | TURN/coturn relay `turn://user:pass@host:port?transport=udp` | placeholder |
| `VIDEO_SOURCE` | `test` (synthetic test pattern, no camera needed) \| `v4l2` (real camera) | `test` |
| `CAMERAS` | Which streams to open (see below) | `cam0,cam1` |
| `WIDTH` / `HEIGHT` | Capture resolution | `640` / `480` |
| `FPS` | Default frame rate (the `@fps` of each entry in `CAMERAS` overrides it) | `30` |
| `ENCODER` | Sender encoder: `auto` \| `nv` (NVENC) \| `sw` (x264, no GPU needed) | `auto` |
| `DECODER` | Receiver decoder: `auto` \| `nv` (NVDEC) \| `sw` (avdec_h264) | `auto` |
| `FLASK_HOST` / `FLASK_PORT` | Receiver HTTP service listen address/port (preview + buffer endpoints) | `127.0.0.1` / `5082` |
| `LK_URL/LK_KEY/LK_SECRET` | **Optional**, only for the LiveKit comparison benchmark; not needed to run LiOS | empty |

`CAMERAS` format:
- `test` mode: comma-separated stream names, optionally with a frame rate → `cam0,cam1` or `cam0@30,cam1@15`
- `v4l2` mode: `name=/dev/videoN@fps` → `mid=/dev/video0@30,left=/dev/video4@25`

What `auto` means: `ENCODER=auto` uses NVENC when it detects `nvh264enc`+`cudaupload`, otherwise it automatically falls back to
x264 software encoding; `DECODER=auto` works the same way (NVDEC via `nvh264dec` if available, otherwise `avdec_h264`).

---

## 4. Run it end-to-end

### 4.1 Single-machine quick check (the easiest way; confirm the link first)

Open three terminals on the same machine (all in the repo root). The default `.env` is fine (`SIGNAL_URL` points to localhost).

```bash
# Terminal 1 —— signal server
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080

# Terminal 2 —— receiver (start first, wait for the stream)
pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2

# Terminal 3 —— sender (videotestsrc test pattern)
pixi run python examples/two_cemera_sender.py
```

> If you'd rather not configure TURN on a single machine, leave the `TURN` placeholder in `.env`—localhost can usually connect directly via host candidates.
> Across machines you almost always need a real, reachable TURN.

### 4.2 Cross-machine deployment (three roles)

It's recommended to put **signal-server and receiver on the same cloud machine (machine B)** and the sender on the edge machine (machine A).
Let machine B's reachable IP be `B_IP`.

**Machine B (cloud): run signal-server + receiver**
Key `.env` items:
```bash
ROOM=demo
SIGNAL_URL=ws://127.0.0.1:18080/ws     # signal-server is local on B
TURN=turn://USER:PASS@TURN_HOST:3478?transport=udp
FLASK_HOST=0.0.0.0                      # set 0.0.0.0 to view the preview from another machine
```
```bash
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080   # terminal 1
pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2                # terminal 2
```

**Machine A (edge): run sender**
Key `.env` items (`SIGNAL_URL` must point to B_IP):
```bash
ROOM=demo
SIGNAL_URL=ws://B_IP:18080/ws          # ← the key part
TURN=turn://USER:PASS@TURN_HOST:3478?transport=udp
VIDEO_SOURCE=test
```
```bash
pixi run python examples/two_cemera_sender.py
```

**Two things that must be reachable across machines:**
1. Machine A → `B_IP` on **TCP 18080** (signaling).
2. A, B → the TURN host on **UDP 3478** (media relay).

**When 18080 can't be reached directly (only an SSH channel, e.g. a Coder workspace): use SSH port forwarding**

Without changing `.env` (both ends still use `127.0.0.1:18080`), tunnel local 18080 on **A (the sender machine)**
to B's signal-server:
```bash
ssh -N -L 18080:127.0.0.1:18080 <RECEIVER_SSH_HOST>
# Replace <RECEIVER_SSH_HOST> with the name you SSH into B (the receiver machine) with, e.g. user@1.2.3.4 or an alias from ~/.ssh/config
```
Keep this terminal open; while the tunnel is alive, `ws://127.0.0.1:18080/ws` on A reaches B's signal-server directly.
**Only this one signaling port needs forwarding**—the media goes over TURN (the `?transport=udp` host) and never through the tunnel.
While you're at it, add the preview port too (`-L 5082:127.0.0.1:5082`) and you can watch B's live feed in your local browser.

### 4.3 How to tell it's working

Check the logs in this order:
- **signal-server**: `listening on :18080` → after both ends connect, `joined room=demo peer=sender-xxx` / `peer=receiver-xxx`.
- **sender**: `[sender] source=test encoder=nv/sw ...` → `[webrtc] sent offer` → `[webrtc] connection state: connected`.
- **receiver**: `[receiver] decoder=...` → `[receiver] appsink bound: cam0` → a continuous stream of
  `[cam0] frame (224, 224, 3) from appsink` = **end-to-end is working**.
- The receiver also prints `[flask] serving on http://HOST:5082 ...`.

### 4.4 Configuring coturn (practically mandatory across machines)

On a single machine you don't need TURN (it connects directly via host candidates). **As soon as you go cross-machine/cross-NAT**, the two ends'
host candidates can't reach each other, and you need a TURN relay to forward the media. The core LiOS approach is to deploy this coturn **close to the cloud
GPU (machine B)** so the media takes the shortest detour.

The code consumes exactly **one** TURN URI (`webrtcbin`'s `turn-server` property) using long-term credentials
(username/password), so configuring a single coturn user is enough.

**1) Install (Ubuntu/Debian):**
```bash
sudo apt-get install -y coturn
```

**2) Write `/etc/turnserver.conf` (minimal working config):**
```ini
listening-port=3478
fingerprint
lt-cred-mech                       # long-term credential mechanism, paired with user= below
user=lios:S3cret-change-me         # username:password, must match the TURN in .env
realm=lios
external-ip=B_PUBLIC_IP            # required when the relay is behind NAT/cloud: its public/reachable IP
min-port=49152                     # relay port range, open it in the firewall too
max-port=65535
no-tls                             # no TLS (turns://); plain turn:// only
no-dtls
no-cli
```

**3) Start the service:**
```bash
# Option A: systemd (first let the default config allow startup)
echo 'TURNSERVER_ENABLED=1' | sudo tee /etc/default/coturn
sudo systemctl enable --now coturn

# Option B: run in the foreground (most direct for watching logs while debugging)
turnserver -c /etc/turnserver.conf -v
```

**4) Open the firewall (critical—miss this and the sender stays stuck at `connecting`):**
- `UDP/TCP 3478` —— the TURN control port
- `UDP 49152-65535` —— the relay media port range (aligned with `min/max-port` above)

**5) Put the same line in both ends' `.env` (matching the `user=` credentials and the relay machine IP):**
```bash
TURN=turn://lios:S3cret-change-me@B_PUBLIC_IP:3478?transport=udp
```

**6) Verify TURN itself works (independent of the whole video pipeline):**
```bash
# coturn's built-in load-test client; getting a relay address means credentials + ports are correct
turnutils_uclient -v -u lios -w S3cret-change-me B_PUBLIC_IP
# Or open a Trickle ICE page in the browser, enter this turn:// and see if you get a typ relay candidate
```
Once the sender log reaches `connection state: connected` and the receiver keeps printing `frame ... from appsink`,
the whole chain—including TURN—is working.

> Note: `transport=udp` refers only to the sender/receiver **to TURN** hop using UDP; coturn listens on 3478 over both
> UDP and TCP by default. If a client's network only allows TCP, change the URI to `?transport=tcp` (and make sure 3478/TCP is open).

---

## 5. Using a real camera (v4l2)

1. First, check which devices and formats are available:
   ```bash
   ls /dev/video*
   v4l2-ctl --list-devices                 # if v4l-utils is installed
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```
2. In the (sender machine's) `.env`:
   ```bash
   VIDEO_SOURCE=v4l2
   CAMERAS=mid=/dev/video0@30,left=/dev/video4@25
   WIDTH=640
   HEIGHT=480
   ```
3. Start the sender: `pixi run python examples/two_cemera_sender.py`

**Things to watch out for:**
- The stream names (`mid`/`left`) are passed to the receiver as the msid; the receiver uses them as the key for the inference buffer and also
  for selecting via `?cam=` in the HTTP preview.
- The example's v4l2 pipeline requests `format=YUY2`. If your camera only supports MJPG/another format, this will fail;
  you'll need to adjust the caps in `v4l2_source()` in `examples/two_cemera_sender.py` accordingly (change `format=YUY2`
  to what the camera supports, or insert `jpegdec`).
- The resolution/frame rate must be a combination the camera actually supports (confirm with `--list-formats-ext` above).

---

## 6. Watching the live feed over HTTP

The receiver starts an HTTP service in the background (`FLASK_HOST:FLASK_PORT`, default `127.0.0.1:5082`) that exposes three endpoints:

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/preview` | **Live MJPEG video stream**; open it in a browser to watch the live feed directly |
| `GET /api/v1/preview.jpg` | A single JPEG of the latest frame (good for scripted snapshots/manual refresh) |
| `GET /api/v1/infer-buffer/base64` | The raw inference buffer (base64, for a same-machine process to fetch data; not an image) |
| `GET /api/v1/healthz` | Health check |

**Watch the live feed:** open in a browser
```
http://<接收端IP>:5082/api/v1/preview
```
With multiple streams, select one / limit the frame rate:
```
http://<接收端IP>:5082/api/v1/preview?cam=cam0&fps=15
```

**To view from another machine:** set `FLASK_HOST=0.0.0.0` in `.env` before starting the receiver (the default `127.0.0.1` only allows local viewing),
and open the port. **If you only have an SSH channel** (e.g. a Coder workspace), keep the default `127.0.0.1` and just forward it locally:
```bash
ssh -N -L 5082:127.0.0.1:5082 <RECEIVER_SSH_HOST>   # locally open http://127.0.0.1:5082/api/v1/preview
```

**Can't open it? First diagnose with curl, bypassing the proxy** (separating "does the service/link have data" from "the browser is misconfigured"):
```bash
curl -sS --noproxy '*' -o /tmp/p.jpg -w "HTTP %{http_code} %{size_download}B\n" \
  http://127.0.0.1:5082/api/v1/preview.jpg
```
- `HTTP 200` with a few KB → the service is fine, the problem is the **browser proxy** (add `localhost,127.0.0.1` to the no-proxy whitelist).
- `HTTP 404` / `0B` → the service is up but **has no frames yet**; go back to the receiver and check whether it's printing `frame ... from appsink`.
- `Couldn't connect` → Flask isn't up or the port/tunnel is wrong; go back to the receiver and look for `[flask] serving on ...`.

**Security reminder:** the preview and buffer endpoints have **no authentication**; `FLASK_HOST=0.0.0.0` exposes the feed to everyone on that network segment.
Only do this on a trusted internal network; add a reverse proxy/authentication for public networks.

> How it works: `/preview` takes the latest frame from the live inference buffer (an HWC uint8 RGB tensor), copies it to the CPU under a lock,
> encodes it as JPEG, and returns it. `/api/v1/infer-buffer/base64` uses CUDA-IPC serialization and **can't be restored to an image across machines**
> (the handle is only valid on the same machine and GPU), so for remote "watch the feed" use `/preview`.

---

## 7. No GPU (running on CPU only)

The whole chain can run on a machine without NVENC/NVDEC:

- **Sender**: set `ENCODER=sw` in `.env` (or rely on `auto` to detect it) to use `x264enc` software encoding.
- **Receiver**: set `DECODER=sw` in `.env` (or `auto`) to use `avdec_h264` software decoding.
- **Inference buffer**: without CUDA it automatically uses a CPU tensor (works fine, this part just isn't CUDA-IPC zero-copy anymore).

```bash
# .env on a machine without a GPU
ENCODER=sw
DECODER=sw
```

> Tip: some machines have the `nvh264enc` plugin installed but can't open an NVENC session at runtime. `auto` usually detects this correctly
> (it fails to register and falls back to software encoding automatically); if `auto` misjudges, setting `ENCODER=sw` explicitly is the most reliable.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| signal-server has no `joined` log | `SIGNAL_URL` is wrong / the server isn't up / cross-machine port 18080 is blocked (`nc -vz B_IP 18080`) |
| sender handshake reports `InvalidMessage` / `did not receive a valid HTTP response` | **A proxy is enabled locally**: `websockets` (v15+) routes even `ws://127.0.0.1` through `ALL_PROXY`/`HTTPS_PROXY`, breaking the handshake. Run `export no_proxy=localhost,127.0.0.1` (or temporarily disable the proxy) and restart |
| sender stuck at `connection state: connecting`/`failed` | ICE didn't connect, almost always **TURN**: wrong address/credentials, or UDP 3478 is blocked |
| sender reports `nvh264enc`/`cudaupload` not found | this machine lacks the NVIDIA GStreamer plugins → set `ENCODER=sw` |
| receiver connects but no `frame ... from appsink` | the stream isn't arriving or decoding failed; check TURN/network first; set `GST_DEBUG=4` for details |
| browser opens `/preview` to 404 / no frame | no frames received yet (wait for the sender to connect), or the `?cam=` name doesn't exist |
| `curl` returns an image but the browser can't open the preview | the **browser proxy** forwards `127.0.0.1`; add `localhost,127.0.0.1` to the browser's "no proxy" whitelist |
| can't open the preview remotely | the receiver's `FLASK_HOST` is still `127.0.0.1`; change it to `0.0.0.0` and open the port; or forward 5082 over SSH (see below) |
| v4l2 fails to start | the camera doesn't support the requested `WIDTH/HEIGHT/fps/YUY2` combination; verify with `v4l2-ctl --list-formats-ext` |

To debug pipeline details: add `GST_DEBUG=4` (or higher) on either end before starting.

---

## 9. Appendix: command cheat sheet

```bash
# signal server
cd signal-server && go build -o webrtcssvr . && ./webrtcssvr serve --addr :18080

# receiver (N streams, listening on 5082 on all interfaces so the preview is reachable remotely)
FLASK_HOST=0.0.0.0 pixi run python examples/two_cemera_receiver_inferbuf.py --streams 2

# sender (test pattern / real camera, decided by VIDEO_SOURCE in .env)
pixi run python examples/two_cemera_sender.py

# watch the live feed in a browser
#   http://<receiver IP>:5082/api/v1/preview
```

For more on design/architecture, see [`design.md`](design.md) and [`docs/`](docs/).
