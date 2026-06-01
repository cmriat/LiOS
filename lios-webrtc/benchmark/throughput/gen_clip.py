"""生成一段固定的合成测试片段(raw I420), 供 gst 与 LiveKit 两端编码"同一段"。
确定性(seed 固定) + 真实熵(平滑纹理+运动+适度噪声), 不是空帧/静止图——这样编码器会产出
真实大小的帧, 高 fps 下网络才会被真正压上, localhost 与跨网络才有区别。

Env: W H FRAMES CLIP
Run: pixi run python benchmark/throughput/gen_clip.py
布局: 每帧 = Y(W*H) + U(W/2*H/2) + V(W/2*H/2), 顺序写入。
"""

import os
import numpy as np

W = int(os.environ.get("W", 640))
H = int(os.environ.get("H", 480))
N = int(os.environ.get("FRAMES", 300))
OUT = os.environ.get("CLIP", "/tmp/clip_i420.raw")

rng = np.random.default_rng(42)
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
yc, xc = np.mgrid[0 : H // 2, 0 : W // 2].astype(np.float32)

with open(OUT, "wb") as f:
    for n in range(N):
        ph = n * 0.20  # 运动相位
        Y = (
            128
            + 50 * np.sin(xx / 17.0 + ph)
            + 40 * np.cos(yy / 13.0 - ph * 0.7)
            + 30 * np.sin((xx + yy) / 29.0 + ph * 1.3)
        )
        Y += rng.normal(0, 6, (H, W))  # 适度噪声 -> 真实熵, 编码器不能压成 0
        U = 128 + 30 * np.sin(xc / 20.0 + ph * 0.5)
        V = 128 + 30 * np.cos(yc / 20.0 - ph * 0.5)
        for plane in (Y, U, V):
            f.write(np.clip(plane, 0, 255).astype(np.uint8).tobytes())

size = os.path.getsize(OUT)
print(f"wrote {OUT}: {N} frames {W}x{H} I420, {size / 1e6:.1f} MB, {size // N} B/frame")
