#!/usr/bin/env python3
"""修图结果评估 — 对拍集 PSNR/SSIM + 全案例无参指标 + Owner 总览拼图。

评分说明（AI 代理评分，1–5，待 Owner 目测校准）：
  修复自然度：对拍集按 SSIM 分档；无 GT 案例按 噪声下降 + 清晰度增益 综合
  细节：输出 Laplacian 方差分档
  色彩：对拍集按与 GT 色彩度差；无 GT 按色彩度绝对分档
  干净度：输出噪声 sigma 分档（越低越干净）
  商用可用度：0.35*修复 + 0.25*细节 + 0.2*干净 + 0.2*色彩，长边 ≥1600 加 0.5（封顶 5）

CLI:
    python3 inference/photo/evaluate.py --run <output_dir>/run.json
        [--pairs <materials>/pairs] [--overview <output_dir>/owner_overview.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# ------------------------------------------------------------------ 指标
def _gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def sharpness(img_rgb: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(img_rgb), cv2.CV_64F).var())


def noise_sigma(img_rgb: np.ndarray) -> float:
    g = _gray(img_rgb)
    med = cv2.medianBlur(g, 3)
    diff = g.astype(np.float32) - med.astype(np.float32)
    return float(np.median(np.abs(diff)) * 1.4826)


def colorfulness(img_rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk colorfulness metric."""
    r, g, b = img_rgb[..., 0].astype(np.float32), img_rgb[..., 1].astype(np.float32), img_rgb[..., 2].astype(np.float32)
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def basic_stats(img_rgb: np.ndarray) -> dict:
    g = _gray(img_rgb)
    return {
        "sharpness_lapvar": round(sharpness(img_rgb), 1),
        "noise_sigma": round(noise_sigma(img_rgb), 2),
        "colorfulness": round(colorfulness(img_rgb), 1),
        "brightness": round(float(g.mean()), 1),
        "clip_dark_pct": round(float((g <= 2).mean() * 100), 2),
        "clip_white_pct": round(float((g >= 253).mean() * 100), 2),
    }


def _read_rgb(p: str | Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def _resize_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)


def _resize_long(img: np.ndarray, long_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_LANCZOS4)


# ------------------------------------------------------------------ 评分
def _band(v: float, bands: list[tuple[float, float]]) -> float:
    """bands: [(阈值, 分数)] 从低到高检查，v <= 阈值 取该档分数。"""
    for th, score in bands:
        if v <= th:
            return score
    return bands[-1][1]


def score_case(m: dict, b: dict, a: dict, ssim_val: float | None) -> dict:
    # 修复自然度
    if ssim_val is not None:
        repair = _band(ssim_val, [(0.45, 1), (0.55, 2), (0.65, 3), (0.75, 4), (1.01, 5)])
    else:
        nr = (b["noise_sigma"] - a["noise_sigma"]) / max(b["noise_sigma"], 1e-6)
        sg = a["sharpness_lapvar"] / max(b["sharpness_lapvar"], 1e-6)
        pts = 3 + (1 if nr > 0.35 else 0) + (1 if sg > 1.2 else 0) - (1 if nr < 0.05 else 0)
        repair = float(max(1, min(5, pts)))
    # 细节
    detail = _band(a["sharpness_lapvar"], [(20, 1), (50, 2), (120, 3), (260, 4), (1e9, 5)])
    # 干净度
    clean = _band(a["noise_sigma"], [(1.5, 5), (3.0, 4), (5.0, 3), (8.0, 2), (1e9, 1)])
    # 色彩
    if ssim_val is not None and m.get("gt"):
        gt_arr = _read_rgb(m["gt"])
        cf_gt = colorfulness(gt_arr)
        color = float(max(1, min(5, 5 - abs(a["colorfulness"] - cf_gt) / 12.0)))
    else:
        cf = a["colorfulness"]
        color = _band(cf, [(4, 1), (12, 2), (22, 3), (42, 4), (1e9, 5)])
    commercial = 0.35 * repair + 0.25 * detail + 0.2 * clean + 0.2 * color
    if max(m.get("sizes", {}).get("after", [0, 0])) >= 1600:
        commercial = min(5, commercial + 0.5)
    return {
        "修复自然度": round(repair, 1), "细节": round(detail, 1),
        "色彩": round(color, 1), "干净度": round(clean, 1),
        "商用可用度": round(commercial, 1),
    }


# ------------------------------------------------------------------ 总览图
def _font(path: str, size: int):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _render_section(items: list[dict], width: int, cols: int, title: str) -> Image.Image | None:
    """把一个分区（交付样例 / 评测附录）渲染成网格图。"""
    thumbs = []
    for m in items:
        im = Image.open(m["outputs"]["compare"]).convert("RGB")
        s = width / im.width
        im = im.resize((width, int(im.height * s)), Image.LANCZOS)
        thumbs.append((m["case"], im, m))
    if not thumbs:
        return None
    rows = (len(thumbs) + cols - 1) // cols
    col_w = width + 24
    row_h = max(t.height for _, t, _ in thumbs) + 76
    canvas = Image.new("RGB", (cols * col_w + 24, rows * row_h + 70), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    fb = _font(FONT_BOLD, 30)
    fr = _font(FONT_REGULAR, 22)
    d.text((24, 16), title, font=_font(FONT_BOLD, 34), fill=(20, 20, 20))
    y0 = 64
    for i, (name, im, m) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = 24 + c * col_w
        y = y0 + r * row_h
        d.text((x, y), name, font=fb, fill=(0, 80, 160))
        canvas.paste(im, (x, y + 42))
        st = m.get("timings", {}).get("total")
        d.text((x + 12, y + 46 + im.height), f"total {st}s  ·  {m.get('mode')}", font=fr, fill=(100, 100, 100))
    return canvas


def build_overview(cases: list[dict], out_path: Path, width: int = 760) -> None:
    """Owner 总览：**只把交付样例放正文**（老照片 hist / 电商 prod）；对拍集（pair*）
    单列「评测附录」并标注用途（内部回归基线，非交付样例）。"""
    done = [m for m in cases if m["status"] == "done"]
    delivery = [m for m in done if not m["case"].startswith("pair")]
    pairs = [m for m in done if m["case"].startswith("pair")]
    sections = [
        ("交付样例（老照片 hist / 电商 prod）", delivery),
        ("评测附录（对拍集 pair · 内部回归基线，有 GT 算 PSNR/SSIM，非交付样例）", pairs),
    ]
    imgs = [img for img in (_render_section(items, width, 2, t) for t, items in sections) if img]
    if not imgs:
        return
    gap = 24
    W = max(i.width for i in imgs)
    H = sum(i.height for i in imgs) + gap * (len(imgs) - 1) + 62
    canvas = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    d.text((24, 14), "AI 修图产线 v0 — Owner 总览（上：输入 / 下：输出）",
           font=_font(FONT_BOLD, 38), fill=(20, 20, 20))
    y = 62
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.height + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"[overview] 交付 {len(delivery)} + 评测附录 {len(pairs)} -> {out_path}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="修图结果评估")
    ap.add_argument("--run", required=True, help="run.json 路径")
    ap.add_argument("--out", default=None, help="metrics.json 输出（默认 run.json 同目录 metrics.json）")
    ap.add_argument("--overview", default=None, help="Owner 总览拼图输出路径")
    args = ap.parse_args()

    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    cases = run["cases"]
    results = []
    for m in cases:
        if m["status"] != "done":
            results.append({"case": m["case"], "mode": m["mode"], "status": m["status"],
                            "error": m.get("error", "")})
            continue
        before = _read_rgb(m["outputs"]["before"])
        after = _read_rgb(m["outputs"]["after"])
        b, a = basic_stats(before), basic_stats(after)
        ssim_val = None
        psnr_val = None
        psnr_1k = ssim_1k = None
        if m.get("gt"):
            gt = _read_rgb(m["gt"])
            gt_c = _resize_to(gt, (after.shape[1], after.shape[0]))
            psnr_val = round(float(psnr(gt_c, after, data_range=255)), 2)
            ssim_val = round(float(ssim(gt_c, after, channel_axis=2, data_range=255)), 4)
            # 统一 1024 长边口径（消除输出分辨率差异，便于 AI 修复 vs ESRGAN 基线对比）
            a1k = _resize_long(after, 1024)
            g1k = _resize_to(gt, (a1k.shape[1], a1k.shape[0]))
            psnr_1k = round(float(psnr(g1k, a1k, data_range=255)), 2)
            ssim_1k = round(float(ssim(g1k, a1k, channel_axis=2, data_range=255)), 4)
        scores = score_case(m, b, a, ssim_val)
        results.append({
            "case": m["case"], "mode": m["mode"], "status": "done",
            "timings": m.get("timings", {}),
            "sizes": m.get("sizes", {}),
            "strength": m.get("strength"), "seed": m.get("seed"),
            "psnr_vs_gt": psnr_val, "ssim_vs_gt": ssim_val,
            "psnr_vs_gt_1024": psnr_1k, "ssim_vs_gt_1024": ssim_1k,
            "before": b, "after": a, "scores": scores,
        })

    out = Path(args.out) if args.out else Path(args.run).with_name("metrics.json")
    summary = {
        "run": run.get("run"),
        "cases": len(cases),
        "ok": sum(1 for r in results if r["status"] == "done"),
        "fail_rate": run.get("summary", {}).get("fail_rate"),
        "stage_avg_seconds": run.get("summary", {}).get("stage_avg_seconds"),
        "results": results,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))

    overview = args.overview or str(Path(args.run).parent / "owner_overview.png")
    build_overview(cases, Path(overview))


if __name__ == "__main__":
    main()
