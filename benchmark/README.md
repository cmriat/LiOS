# Benchmarks

Reproduces the figures in the [performance report](../docs/gst-report/) and [README](../README.md):
throughput (achieved vs target fps), delivery jitter, one-way latency, and the architecture diagram.

All cross-stack comparisons use **identical content** (one synthetic clip encoded on both sides),
**matched bitrate**, and **per-frame counting at the receiver** (a direct decoded-frame count, not the
`fpsdisplaysink` rate signal, which over-reports for this stream).

## Figures

```bash
pixi run python benchmark/make_figures.py [arch|latency|throughput|jitter|all]   # default: all
```

| Figure | Output | Data source |
|---|---|---|
| `arch` | `docs/gst-report/arch_imgtx.png` | self-contained |
| `latency` | `docs/gst-report/latency_compare.png` | `relat_results/*.txt` |
| `throughput` | `docs/gst-report/throughput_samecontent.png` | embedded measured points |
| `jitter` | `docs/gst-report/throughput_jitter_violin.png` | `/tmp/{gst,lk}_ivals.txt` (DUMP) |

## Layout

```
benchmark/
  make_figures.py                 # all four figures (subcommand-selectable)
  throughput/                     # gst-webrtc throughput
    gen_clip.py                   #   generate the shared synthetic clip -> /tmp/clip_i420.raw
    gst_sender_clip.py            #   appsrc(clip) -> NVENC -> RTP
    gst_receiver_appsink.py       #   reliable per-frame count (appsink, no GIL-inflation)
    gst_sender.py / gst_receiver.py / sweep_gst.sh   # videotestsrc pipeline-ceiling sweep (~1685 fps)
    common.py                     #   FpsMeter / FpsCollector helpers
  livekit/                        # LiveKit (libwebrtc) throughput + latency
    livekit_publisher.py          #   publish clip (CLIP env, MAXFPS, MAXBR)
    livekit_subscriber.py         #   reliable per-frame count (+ DUMP intervals)
    common.py                     #   FpsMeter (kept local so this dir is self-contained / scp-able)
    cloud_pub.sh                  #   admin01 -> LiveKit Cloud publisher (cross-machine)
    remote_sub.sh                 #   remote subscriber (scp this dir to the remote host)
    livekit_sender_ts.py / livekit_receiver_ts.py    # latency clients (1 fps timestamped)
    cross_lk_latency_bidir.sh / run_lat.sh           # LiveKit latency (NTP 4-timestamp)
    run_rust_throughput.sh / lk_rust/                # native Rust libwebrtc throughput
  cross_p2p_latency_bidir.sh      # gst latency (NTP 4-timestamp) -> relat_results/p2p.txt
  cross_p2p_sweep.sh              # gst cross-machine throughput (admin01 -> edge relay -> remote)
  relat_repeat.sh                 # repeat a latency measurement N runs (-> distribution)
  two_camera_sender_1fps_ts.py / two_camera_receiver_inferbuf_1fps_ts.py   # gst latency clients
  relat_results/                  # latency result files consumed by make_figures.py latency
```

## Reproduce (localhost, same-content throughput)

```bash
pixi run python benchmark/throughput/gen_clip.py        # -> /tmp/clip_i420.raw (run once)

# gst: clip @ target fps, reliable count (dumps inter-frame intervals for the jitter violin)
ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws FPS=400 BITRATE_KBPS=10000 CLIP=/tmp/clip_i420.raw \
  pixi run python benchmark/throughput/gst_sender_clip.py &
ROOM=demo SIGNAL_URL=ws://127.0.0.1:18080/ws DURATION=12 WARMUP=4 DUMP=/tmp/gst_ivals.txt \
  pixi run python benchmark/throughput/gst_receiver_appsink.py

# LiveKit: same clip, matched bitrate, reliable count (DUMP for the violin)
#   subscriber first, then publisher (FPS=MAXFPS, MAXBR=10000000, CLIP=/tmp/clip_i420.raw)
#   see livekit/livekit_subscriber.py / livekit_publisher.py
```

Cross-machine LiveKit (admin01 → LiveKit Cloud → remote): see `livekit/cloud_pub.sh` (publisher,
reads `.env` Cloud creds) and `livekit/remote_sub.sh` (subscriber; scp the `livekit/` dir to the remote host).
