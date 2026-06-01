#!/usr/bin/env python3
"""Generate LiOS benchmark figures (all English, for the report / README).

Usage:
    pixi run python benchmark/make_figures.py [FIGURE]
    FIGURE = arch | latency | throughput | jitter | all   (default: all)

Outputs -> docs/gst-report/:
    arch        -> arch_imgtx.png            (architecture diagram)
    latency     -> latency_compare.png       (one-way latency, reads benchmark/relat_results/*.txt)
    throughput  -> throughput_samecontent.png(achieved vs target fps, same clip + matched bitrate)
    jitter      -> throughput_jitter_violin.png (inter-frame interval, reads /tmp/{gst,lk}_ivals.txt)
"""

import argparse
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(os.path.dirname(HERE), "docs", "gst-report")
RESDIR = os.path.join(HERE, "relat_results")
GST, LK = "#1f9e89", "#d9534f"


# ---------------------------------------------------------------- throughput
def fig_throughput():
    gst_x = [100, 200, 400, 600, 800, 1000, 1200]
    gst_y = [100, 200, 402, 604, 804, 1008, 1202]
    lk_x = [100, 200, 400, 600, 800]
    lk_y = [100, 200, 400, 561, 692]
    LK_CEIL, LK_BREAK = 700, 1000

    fig, ax = plt.subplots(figsize=(8.6, 6.2), dpi=160)
    ax.plot([0, 1260], [0, 1260], ls=(0, (4, 4)), color="#9aa0a6", lw=1.2, label="achieved = target (ideal)", zorder=1)
    ax.plot(gst_x, gst_y, "-o", color=GST, lw=2.4, ms=6.5, label="LiOS (NVENC/NVDEC)", zorder=3)
    ax.plot(lk_x, lk_y, "-s", color=LK, lw=2.4, ms=6.5, label="LiveKit (libwebrtc)", zorder=3)
    ax.plot([lk_x[-1], LK_BREAK], [lk_y[-1], 0], ls=(0, (2, 2)), color=LK, lw=2.0, zorder=2)
    ax.scatter([LK_BREAK], [0], marker="x", s=150, color=LK, lw=3, zorder=4)
    ax.axhline(LK_CEIL, ls=":", color=LK, lw=1.3, alpha=0.85)
    ax.text(60, LK_CEIL + 18, f"LiveKit saturation ≈ {LK_CEIL} fps", color=LK, fontsize=10.5, fontweight="bold")
    ax.annotate(
        "LiOS: no saturation up to 1200 fps\n(receiver GIL limit, not pipeline limit)",
        xy=(1200, 1202),
        xytext=(520, 1095),
        fontsize=10.5,
        color="#0f5e52",
        fontweight="bold",
    )
    ax.annotate(
        f"≥{LK_BREAK} fps: subscriber stream interrupted",
        xy=(LK_BREAK, 0),
        xytext=(610, 95),
        fontsize=10.5,
        color=LK,
        fontweight="bold",
    )
    ax.set_xlabel("Target frame rate (fps)", fontsize=12)
    ax.set_ylabel("Achieved frame rate (fps)", fontsize=12)
    ax.set_xlim(0, 1260)
    ax.set_ylim(0, 1300)
    ax.grid(ls=(0, (3, 4)), alpha=0.35)
    ax.set_title(
        "Single-stream decode throughput: achieved vs. target\n"
        "identical clip · 10 Mbps · localhost · per-frame counting at receiver",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
    fig.text(
        0.5,
        0.013,
        "Single-stream ceiling:  LiOS ≥ 1200 fps (unsaturated)   |   "
        "LiveKit ≈ 700 fps (saturated), stream drops at ≥ 1000 fps\n"
        "Same synthetic clip encoded on both sides · 10 Mbps · per-frame counting (not fpsdisplaysink signal) "
        "· gst pipeline ceiling ≈ 1685 fps (videotestsrc, measured separately)",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#777",
    )
    fig.subplots_adjust(bottom=0.15, top=0.89)
    out = os.path.join(OUTDIR, "throughput_samecontent.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- jitter
def fig_jitter():
    def load(p):
        return np.array([float(x) for x in open(p) if x.strip()])

    gp, lp = "/tmp/gst_ivals.txt", "/tmp/lk_ivals.txt"
    if not (os.path.exists(gp) and os.path.exists(lp)):
        print(f"[skip jitter] need {gp} and {lp} (DUMP from gst_receiver_appsink.py / livekit_subscriber.py)")
        return
    g, l = load(gp), load(lp)
    TARGET_FPS = 400
    IDEAL = 1000.0 / TARGET_FPS
    ymax = max(np.percentile(g, 99), np.percentile(l, 99)) * 1.4
    gc, lc = g[g <= ymax], l[l <= ymax]

    fig, ax = plt.subplots(figsize=(7.8, 6.0), dpi=160)
    parts = ax.violinplot([gc, lc], positions=[1, 2], showmedians=True, showextrema=True, widths=0.8)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([GST, LK][i])
        pc.set_alpha(0.55)
        pc.set_edgecolor([GST, LK][i])
    for k in ("cmedians", "cmaxes", "cmins", "cbars"):
        if k in parts:
            parts[k].set_edgecolor("#333")
            parts[k].set_linewidth(1.2)
    ax.axhline(IDEAL, ls="--", color="#888", lw=1.3)
    ax.text(2.5, IDEAL, f"  ideal {IDEAL:.1f} ms", va="center", ha="left", fontsize=10, color="#666")
    for x, d, c in ((1, g, GST), (2, l, LK)):
        ax.text(
            x,
            ymax * 0.96,
            f"p50 = {np.percentile(d, 50):.2f} ms\np95 = {np.percentile(d, 95):.2f} ms\nstd = {d.std():.2f} ms",
            ha="center",
            va="top",
            fontsize=10,
            color=c,
            fontweight="bold",
        )
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["LiOS", "LiveKit"], fontsize=12.5)
    ax.set_ylabel("Inter-frame arrival interval (ms)", fontsize=12)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", ls=(0, (3, 4)), alpha=0.35)
    ax.set_title(
        "Inter-frame arrival interval · identical clip @400 fps · localhost\n"
        "ideal = 1000/fps = 2.5 ms · tighter distribution ⇒ lower delivery jitter",
        fontsize=12.5,
        fontweight="bold",
        pad=12,
    )
    fig.text(
        0.5,
        0.012,
        f"LiOS n={len(g)} / LiveKit n={len(l)} interval samples · both at ~400 fps mean throughput "
        f"· y-axis clipped near p99 (tail trimmed)",
        ha="center",
        va="bottom",
        fontsize=8.0,
        color="#777",
    )
    fig.subplots_adjust(bottom=0.12, top=0.87)
    out = os.path.join(OUTDIR, "throughput_jitter_violin.png")
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------- latency
def fig_latency():
    SERIES = [
        ("p2p.txt", "LiOS\n(gst + edge relay)", "#2e9e5b", "#1d6e3d"),
        ("lk_sshtunnel.txt", "LiveKit\n(self-hosted + TCP tunnel)", "#4c8bf5", "#2a5dad"),
        ("cloud.txt", "LiveKit Cloud\n(managed, default)", "#d9534f", "#9e342f"),
    ]
    labels, meds, los, his, colors, edges, n_runs = [], [], [], [], [], [], 0
    for fn, lab, c, e in SERIES:
        p = os.path.join(RESDIR, fn)
        if not os.path.exists(p):
            continue
        vals = [float(x) for x in open(p) if x.strip()]
        if not vals:
            continue
        m = st.median(vals)
        labels.append(lab)
        meds.append(m)
        los.append(m - min(vals))
        his.append(max(vals) - m)
        colors.append(c)
        edges.append(e)
        n_runs = len(vals)
        print(f"{fn}: n={len(vals)} median={m:.1f} min={min(vals):.1f} max={max(vals):.1f}")
    if not meds:
        print(f"[skip latency] no result files under {RESDIR}")
        return

    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=160)
    x = range(len(meds))
    ax.bar(x, meds, width=0.56, color=colors, edgecolor=edges, linewidth=1.5, zorder=3)
    ax.errorbar(x, meds, yerr=[los, his], fmt="none", ecolor="#333", elinewidth=1.6, capsize=8, capthick=1.6, zorder=5)
    ymax = max(m + h for m, h in zip(meds, his)) * 1.18
    for xi, (m, h) in enumerate(zip(meds, his)):
        ax.text(
            xi,
            m + h + ymax * 0.02,
            f"≈{round(m)} ms",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            color="#1a1a1a",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("One-way latency (ms)", fontsize=12.5)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", ls="--", alpha=0.35, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.set_title("Camera → cloud CUDA buffer: end-to-end one-way latency", fontsize=15.5, fontweight="bold", pad=16)
    ax.text(
        0.5,
        1.005,
        f"bidirectional NTP method (clock offset cancelled) · {n_runs} runs/group · error bars = measured min–max",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#666",
    )
    fig.subplots_adjust(bottom=0.12, top=0.88)
    out = os.path.join(OUTDIR, "latency_compare.png")
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out, "medians:", [round(m, 1) for m in meds])


# ---------------------------------------------------------------- architecture
def fig_arch():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    from matplotlib.font_manager import FontProperties

    FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    fp = FontProperties(fname=FONT) if os.path.exists(FONT) else FontProperties()
    fpb = FontProperties(fname=FONT, weight="bold") if os.path.exists(FONT) else FontProperties(weight="bold")
    EDGE_BG, EDGE_BD = "#eaf2fb", "#2f6fb0"
    CLOUD_BG, CLOUD_BD = "#eaf7ef", "#2e9e5b"
    RELAY_BG, RELAY_BD = "#fdf3e3", "#d08a2c"
    GPU_BG, GPU_BD = "#fff4f4", "#c0392b"
    SIG_BD = "#6c4fb0"
    fig, ax = plt.subplots(figsize=(15.5, 8.0), dpi=150)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    def zone(x, y, w, h, bg, bd, title, tc):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.18", fc=bg, ec=bd, lw=2.2, zorder=1)
        )
        ax.text(x + w / 2, y + h - 0.34, title, ha="center", va="center", fontproperties=fpb, fontsize=14, color=tc)

    def node(cx, cy, w, h, text, bg="#fff", bd="#888", fs=11, bold=False, tc="#222"):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                fc=bg,
                ec=bd,
                lw=1.5,
                zorder=4,
            )
        )
        ax.text(
            cx,
            cy,
            text,
            ha="center",
            va="center",
            fontproperties=(fpb if bold else fp),
            fontsize=fs,
            color=tc,
            zorder=5,
        )
        return (cx, cy, w, h)

    def varrow(a, b, color="#555", lw=2.0):
        ax.add_patch(
            FancyArrowPatch(
                (a[0], a[1] - a[3] / 2),
                (b[0], b[1] + b[3] / 2),
                arrowstyle="-|>",
                mutation_scale=15,
                lw=lw,
                color=color,
                zorder=3,
            )
        )

    ax.text(
        7.75,
        7.62,
        "LiOS image-transmission architecture",
        ha="center",
        va="center",
        fontproperties=fpb,
        fontsize=21,
        color="#1a1a1a",
    )
    ax.text(
        7.75,
        7.2,
        "End-to-end GPU-resident  ·  NVENC/NVDEC hardware codec  ·  WebRTC real-time transport",
        ha="center",
        va="center",
        fontproperties=fp,
        fontsize=12,
        color="#666",
    )
    ax.add_patch(
        FancyBboxPatch(
            (0.4, 6.25),
            14.7,
            0.62,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc="#f0ecf9",
            ec=SIG_BD,
            lw=1.8,
            zorder=1,
        )
    )
    ax.text(
        7.75,
        6.56,
        "WebSocket signaling  ·  SDP / ICE  ·  multi-camera negotiation",
        ha="center",
        va="center",
        fontproperties=fpb,
        fontsize=12,
        color=SIG_BD,
    )
    ZY, ZH = 0.55, 5.05
    EX, EW = 0.4, 4.4
    CX, CW = 10.7, 4.4
    zone(EX, ZY, EW, ZH, EDGE_BG, EDGE_BD, "Capture / Edge GPU", EDGE_BD)
    zone(CX, ZY, CW, ZH, CLOUD_BG, CLOUD_BD, "Cloud GPU Inference", CLOUD_BD)
    ex, cx = EX + EW / 2, CX + CW / 2
    nw = 3.7
    e1 = node(ex, 4.55, nw, 0.78, "Cameras  cam0 / cam1", bg="#dbe8f7", bd=EDGE_BD, bold=True)
    e2 = node(
        ex, 3.30, nw, 0.82, "NVENC H.264 encode\ncudaupload -> CUDA pool", bg=GPU_BG, bd=GPU_BD, fs=10.5, bold=True
    )
    e3 = node(ex, 1.95, nw, 0.82, "webrtcbin (sendonly)\nRTP · encrypted WebRTC", bg="#dbe8f7", bd=EDGE_BD, fs=10.5)
    varrow(e1, e2)
    varrow(e2, e3)
    c1 = node(cx, 4.55, nw, 0.78, "webrtcbin (recvonly)\nRTP · encrypted WebRTC", bg="#d9efe2", bd=CLOUD_BD, fs=10.5)
    c2 = node(cx, 3.30, nw, 0.82, "NVDEC H.264 decode\nnvvideoconvert · RGBA", bg=GPU_BG, bd=GPU_BD, fs=10.5, bold=True)
    c3 = node(
        cx,
        1.95,
        nw,
        0.82,
        "CUDA memory pool\nCUDA tensor -> VLA inference",
        bg="#d9efe2",
        bd=CLOUD_BD,
        fs=10.5,
        bold=True,
    )
    varrow(c1, c2)
    varrow(c2, c3)
    RX, RW = 5.35, 4.8
    ax.add_patch(
        FancyBboxPatch(
            (RX, 2.85),
            RW,
            1.55,
            boxstyle="round,pad=0.04,rounding_size=0.22",
            fc=RELAY_BG,
            ec=RELAY_BD,
            lw=2.2,
            zorder=1,
        )
    )
    ax.text(
        RX + RW / 2, 4.08, "Public relay", ha="center", va="center", fontproperties=fpb, fontsize=13, color=RELAY_BD
    )
    node(
        RX + RW / 2,
        3.35,
        4.0,
        0.62,
        "coturn  TURN / STUN\nSRTP over UDP",
        bg="#fbe9cf",
        bd=RELAY_BD,
        fs=10.5,
        bold=True,
    )
    ax.add_patch(
        FancyArrowPatch(
            (e3[0] + e3[2] / 2, e3[1] + 0.1),
            (RX + 0.1, 3.55),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.6,
            color="#c0392b",
            connectionstyle="arc3,rad=-0.15",
            zorder=6,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (RX + RW - 0.1, 3.55),
            (c1[0] - c1[2] / 2, c1[1] - 0.1),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.6,
            color="#c0392b",
            connectionstyle="arc3,rad=-0.15",
            zorder=6,
        )
    )
    ax.text(
        RX + RW / 2,
        4.62,
        "encrypted real-time media (SRTP)",
        ha="center",
        va="center",
        fontproperties=fp,
        fontsize=10,
        color="#c0392b",
    )
    for sx, stop in [(ex, e3[1] + e3[3] / 2), (cx, c1[1] + c1[3] / 2)]:
        ax.add_patch(
            FancyArrowPatch(
                (sx, 6.25),
                (sx, stop),
                arrowstyle="-|>",
                mutation_scale=11,
                lw=1.3,
                color=SIG_BD,
                ls=(0, (4, 3)),
                zorder=2,
            )
        )
    ax.text(
        7.75,
        0.18,
        "Red nodes stay resident in GPU memory  -  no CPU<->GPU round-trip",
        ha="center",
        va="center",
        fontproperties=fpb,
        fontsize=11.5,
        color=GPU_BD,
    )
    out = os.path.join(OUTDIR, "arch_imgtx.png")
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out)


FIGS = {"arch": fig_arch, "latency": fig_latency, "throughput": fig_throughput, "jitter": fig_jitter}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate LiOS benchmark figures.")
    ap.add_argument(
        "figure", nargs="?", default="all", choices=["all"] + list(FIGS), help="which figure to generate (default: all)"
    )
    args = ap.parse_args()
    for name in list(FIGS) if args.figure == "all" else [args.figure]:
        FIGS[name]()
