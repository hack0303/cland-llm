#!/usr/bin/env python3
"""Real-ESRGAN 4x 超分（分块推理）— 修复管线第②环。

复用 `inference/sdxl/hand_pipe/rrdbnet.py` 已验证的 RRDBNet 实现（权重命名适配 + 加载校验），
仅在上层增加分块推理：输入大图时按 tile 处理、重叠区线性羽化融合，避免 4x 输出撑爆显存。

CLI:
    python3 inference/photo/upscale.py -i in.jpg -o out.png [--model realesrgan|ultrasharp] [--tile 256]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HP = Path(__file__).resolve().parents[1] / "sdxl" / "hand_pipe"
if str(_HP) not in sys.path:
    sys.path.insert(0, str(_HP))

from rrdbnet import RealESRGANUpscaler  # noqa: E402

MODELS = {
    "realesrgan": "/mnt/data/ai_workspace/models/upscale/RealESRGAN_x4plus.pth",
    "ultrasharp": "/mnt/data/ai_workspace/models/upscale/4x-UltraSharp.pth",
}


class TiledRealESRGAN:
    """4x 超分器：小块直接推理，大图分块 + 羽化融合。"""

    def __init__(self, model: str = "realesrgan", device: str = "cuda",
                 tile: int = 256, overlap: int = 16):
        if model not in MODELS:
            raise ValueError(f"未知超分模型 {model}，可选 {sorted(MODELS)}")
        self.inner = RealESRGANUpscaler(MODELS[model], device=device)
        self.scale = 4
        self.tile = tile
        self.overlap = overlap
        self.device = device
        self.model_name = model

    # ---------------------------------------------------------------- helpers
    def _run(self, img_bgr: np.ndarray) -> np.ndarray:
        out = self.inner.upscale(np.ascontiguousarray(img_bgr))
        return out.astype(np.float32)

    @staticmethod
    def _ramp(n: int) -> np.ndarray:
        """0.05 → 1 → 0.05 的线性窗（与相邻块羽化）。"""
        r = np.ones(n, np.float32)
        if n >= 4:
            r[: n // 2] = np.linspace(0.05, 1.0, n // 2, dtype=np.float32)
            r[n // 2:] = np.linspace(1.0, 0.05, n - n // 2, dtype=np.float32)
        return r

    # ------------------------------------------------------------------- main
    def upscale(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        t, o = self.tile, self.overlap
        s = self.scale
        if h <= t and w <= t:
            return self._run(img_bgr).round().clip(0, 255).astype(np.uint8)

        out = np.zeros((h * s, w * s, 3), np.float32)
        weight = np.zeros((h * s, w * s, 1), np.float32)
        step = max(1, t - o)
        ys = list(range(0, h, step))
        xs = list(range(0, w, step))
        for y in ys:
            for x in xs:
                y2, x2 = min(y + t, h), min(x + t, w)
                y1, x1 = max(0, y2 - t), max(0, x2 - t)
                up = self._run(img_bgr[y1:y2, x1:x2])
                th, tw = up.shape[:2]
                mask = np.ones((th, tw, 1), np.float32)
                ov = min(o * s, th // 2, tw // 2)
                if ov >= 2:
                    ramp = self._ramp(2 * ov)
                    if y1 > 0:                      # 非图像上边界 → 上侧羽化
                        mask[:ov, :, 0] *= ramp[:ov][:, None]
                    if y2 < h:                      # 非图像下边界
                        mask[-ov:, :, 0] *= ramp[-ov:][:, None]
                    if x1 > 0:
                        mask[:, :ov, 0] *= ramp[:ov][None, :]
                    if x2 < w:
                        mask[:, -ov:, 0] *= ramp[-ov:][None, :]
                oy, ox = y1 * s, x1 * s
                out[oy:oy + th, ox:ox + tw] += up * mask
                weight[oy:oy + th, ox:ox + tw] += mask
        out = out / np.maximum(weight, 1e-6)
        return out.round().clip(0, 255).astype(np.uint8)


def upscale_file(up: TiledRealESRGAN, src: str, dst: str) -> dict:
    t0 = time.time()
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读图失败: {src}")
    out = up.upscale(img)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(dst, out)
    return {
        "src": src, "dst": dst,
        "in_size": [img.shape[1], img.shape[0]],
        "out_size": [out.shape[1], out.shape[0]],
        "seconds": round(time.time() - t0, 2),
        "model": up.model_name,
    }


def main():
    ap = argparse.ArgumentParser(description="Real-ESRGAN 4x 超分（分块）")
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--model", default="realesrgan", choices=sorted(MODELS))
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=16)
    args = ap.parse_args()

    up = TiledRealESRGAN(args.model, tile=args.tile, overlap=args.overlap)
    info = upscale_file(up, args.input, args.output)
    print(f"[esrgan] {info['in_size']} -> {info['out_size']}  {info['seconds']}s -> {args.output}")


if __name__ == "__main__":
    main()
