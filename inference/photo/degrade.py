#!/usr/bin/env python3
"""老照片退化模拟 — 由高清 GT 合成"老照片"输入（合成对拍集专用）。

支持的退化算子（--degrade 逗号分隔，按固定顺序执行）：
    fade:0.15        对比度衰减（0~1）
    downscale:3      降采样倍数（LANCZOS，模拟低清扫描）
    blur:1.5         GaussianBlur 半径
    noise:10         高斯噪声 sigma（0~255）
    scratch:6        随机划痕条数（浅色折线）
    frame:0.02       四周暗边/磨损框宽度比例
    sepia:100        sepia 强度 0~100（100=旧着色照片）
    gray:1           转灰度（老式银盐照片≥）
    vignette:0.3     暗角强度（0~1）
    jpeg:80          JPEG 劣化（质量）
执行顺序：fade → downscale → blur → noise → scratch → frame → sepia/gray → vignette → jpeg

CLI:
    python3 inference/photo/degrade.py -i gt.png -o old.jpg \
        --degrade downscale:3,blur:1.2,noise:10,scratch:5,vignette:0.25,jpeg:82
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


def _parse_degrade(spec: str) -> dict:
    ops: dict = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, _, v = part.partition(":")
        ops[k.strip()] = float(v) if v else 1.0
    return ops


def _noise(im: Image.Image, sigma: float, seed: int) -> Image.Image:
    if sigma <= 0:
        return im
    rng = np.random.default_rng(seed)
    arr = np.asarray(im, np.float32)
    arr = arr + rng.normal(0, sigma, arr.shape).astype(np.float32)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _scratches(im: Image.Image, n: int, seed: int) -> Image.Image:
    """随机浅色划痕（折线），模拟胶片/相纸磨损。"""
    if n <= 0:
        return im
    rng = np.random.default_rng(seed)
    w, h = im.size
    layer = Image.new("RGB", (w, h), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(n):
        x = rng.uniform(0, w)
        # 竖向为主，带轻微摆动
        y0 = rng.uniform(0, h * 0.25)
        y1 = rng.uniform(h * 0.6, h)
        pts = []
        yy = y0
        xx = x
        while yy < y1:
            pts.append((xx, yy))
            yy += h * rng.uniform(0.08, 0.2)
            xx += rng.uniform(-w * 0.01, w * 0.01)
        width = int(rng.integers(1, 3))
        val = int(rng.integers(200, 245))
        d.line(pts, fill=(val, val, val), width=width)
    mask = layer.convert("L").filter(ImageFilter.GaussianBlur(0.6))
    return Image.composite(Image.new("RGB", (w, h), (255, 255, 255)), im, mask)


def _frame(im: Image.Image, frac: float) -> Image.Image:
    """四周磨损暗框（模拟相纸边）。"""
    if frac <= 0:
        return im
    w, h = im.size
    arr = np.asarray(im, np.float32)
    b = max(1, int(min(w, h) * frac))
    edge = np.ones((h, w), np.float32)
    ramp = np.linspace(0.55, 1.0, b, dtype=np.float32)
    edge[:b, :] = np.minimum(edge[:b, :], ramp[:, None])
    edge[-b:, :] = np.minimum(edge[-b:, :], ramp[::-1][:, None])
    edge[:, :b] = np.minimum(edge[:, :b], ramp[None, :])
    edge[:, -b:] = np.minimum(edge[:, -b:], ramp[::-1][None, :])
    arr *= edge[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _sepia(im: Image.Image, strength: float) -> Image.Image:
    """银盐/蛋白相纸的暖调着色。strength 0~100。"""
    if strength <= 0:
        return im
    gray = ImageOps.grayscale(im).convert("RGB")
    arr = np.asarray(gray, np.float32)
    sep = np.zeros_like(arr)
    sep[..., 0] = arr[..., 0] * 1.07 + 24   # R
    sep[..., 1] = arr[..., 1] * 0.96 + 12   # G
    sep[..., 2] = arr[..., 2] * 0.78 + 0    # B
    sep = sep.clip(0, 255)
    k = float(strength) / 100.0
    out = arr * (1 - k) + sep * k
    return Image.fromarray(out.astype(np.uint8))


def _vignette(im: Image.Image, strength: float) -> Image.Image:
    if strength <= 0:
        return im
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    mask = 1.0 - strength * (r ** 2)
    arr = np.asarray(im, np.float32) * mask[..., None]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _jpeg(im: Image.Image, quality: int, tmp: Path) -> Image.Image:
    if quality <= 0:
        return im
    tmp.parent.mkdir(parents=True, exist_ok=True)
    im.save(tmp, "JPEG", quality=int(quality))
    return Image.open(tmp).convert("RGB")


def degrade(im: Image.Image, ops: dict, seed: int = 42, tmp_dir: Path | None = None) -> Image.Image:
    if ops.get("fade", 0) > 0:
        im = ImageEnhance.Contrast(im).enhance(max(0.0, 1.0 - ops["fade"]))
    if ops.get("downscale", 0) > 1:
        w, h = im.size
        f = ops["downscale"]
        im = im.resize((max(1, int(w / f)), max(1, int(h / f))), Image.LANCZOS)
    if ops.get("blur", 0) > 0:
        im = im.filter(ImageFilter.GaussianBlur(ops["blur"]))
    if ops.get("noise", 0) > 0:
        im = _noise(im, ops["noise"], seed)
    if ops.get("scratch", 0) > 0:
        im = _scratches(im, int(ops["scratch"]), seed + 1)
    if ops.get("frame", 0) > 0:
        im = _frame(im, ops["frame"])
    if ops.get("gray", 0) > 0:
        im = ImageOps.grayscale(im).convert("RGB")
    if ops.get("sepia", 0) > 0:
        im = _sepia(im, ops["sepia"])
    if ops.get("vignette", 0) > 0:
        im = _vignette(im, ops["vignette"])
    if ops.get("jpeg", 0) > 0:
        im = _jpeg(im, ops["jpeg"], (tmp_dir or Path("/tmp")) / "degrade_tmp.jpg")
    return im


def main():
    ap = argparse.ArgumentParser(description="老照片退化模拟（合成对拍集）")
    ap.add_argument("-i", "--input", required=True, help="GT 高清图")
    ap.add_argument("-o", "--output", required=True, help="退化输出（老照片输入）")
    ap.add_argument("--degrade", required=True,
                    help="退化算子串，如 downscale:3,blur:1.2,noise:10,scratch:5,vignette:0.25,jpeg:82")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ops = _parse_degrade(args.degrade)
    t0 = time.time()
    gt = ImageOps.exif_transpose(Image.open(args.input)).convert("RGB")
    old = degrade(gt, ops, seed=args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    img_format = "PNG" if args.output.lower().endswith(".png") else "JPEG"
    old.save(args.output, img_format, quality=92)
    info = {
        "gt": args.input, "output": args.output,
        "gt_size": list(gt.size), "old_size": list(old.size),
        "ops": ops, "seed": args.seed,
        "seconds": round(time.time() - t0, 2),
    }
    meta = Path(args.output).with_suffix(".degrade.json")
    meta.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[degrade] {gt.size} -> {old.size}  ops={ops}  -> {args.output}")


if __name__ == "__main__":
    main()
