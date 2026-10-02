#!/usr/bin/env python3
"""#225 老照片上色「颜色覆盖位置偏移」四路对比实验（A/B/C/D）。

在真实产线 `inference/photo/retouch.py` 的 colorize 链路上，固定同一输入素材，
对以下四路做 before/after + 手臂/衣物局部放大对拍，并计算客观「偏移/保真」指标：

   A. SDXL 调参    —— 变化 strength / CFG（基线 strength=0.80）
   B. 色度层配准    —— chroma_transfer 前用光流把 SDXL 出色层对齐到 work；再做引导滤波
   C. 掩膜约束      —— DWPose 脸/手/前臂 mask + 躯干/衣物区域，区域色度中值稳定
   D. 专用模型      —— DDColor / DeOldify（本脚本只留接口，另见报告）

用法：
    python3 inference/photo/tools/cmp_colorize_offset.py \
        --input /mnt/data/ai_workspace/outputs/photo-retouch/materials/historical/hist01_migrant_mother.jpg \
        --outdir /mnt/data/ai_workspace/outputs/photo-retouch-225/hist01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import retouch  # noqa: E402

FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# ---------------------------------------------------------------- 客观指标
def _lab(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)


def _grad(m: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def metrics(work: Image.Image, after: Image.Image) -> dict:
    """边缘对齐（越高越好）+ 平滑区串色（越低越好）+ 亮度保真。"""
    wl = _lab(work)
    al = _lab(after)
    L = wl[..., 0]
    gL = _grad(L)
    a = al[..., 1] - 128.0
    b = al[..., 2] - 128.0
    C = np.hypot(a, b)
    gC = _grad(C)
    # 只在高对比结构边缘处评估对齐
    thr = np.percentile(gL, 90)
    edge = gL >= thr
    if edge.sum() > 50:
        x = gL[edge]
        y = gC[edge]
        align = float(np.corrcoef(x, y)[0, 1]) if x.std() > 1e-6 and y.std() > 1e-6 else 0.0
    else:
        align = 0.0
    # 平滑区（低梯度）里色度不应乱变 → 串色
    smooth = gL <= np.percentile(gL, 40)
    bleed = float(gC[smooth].mean() / (gC.mean() + 1e-6))
    # 亮度保真（chroma_transfer 保 L；专用模型会改动 L）
    dL = float(np.mean(np.abs(L - al[..., 0])))
    return {"edge_align": round(align, 4), "bleed_ratio": round(bleed, 4),
            "L_shift": round(dL, 3)}


# ---------------------------------------------------------------- B: 配准
def register_color(color_img: Image.Image, base_img: Image.Image) -> Image.Image:
    """用稠密光流把 color_img 对齐到 base_img（修正局部几何漂移）。"""
    b = np.asarray(base_img.convert("L"))
    c = np.asarray(color_img.convert("L"))
    flow = cv2.calcOpticalFlowFarneback(b, c, None, 0.5, 5, 25, 5, 7, 1.5, 0)
    h, w = b.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mapx = (xx + flow[..., 0]).astype(np.float32)
    mapy = (yy + flow[..., 1]).astype(np.float32)
    warped = cv2.remap(np.asarray(color_img.convert("RGB")), mapx, mapy,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped)


def guided_chroma(im: Image.Image, radius: int = 8, eps: float = 1e-2) -> Image.Image:
    """引导滤波：以 L 为引导平滑 a/b，让色度吸附亮度边缘（减串色）。"""
    lab = _lab(im)
    L = (lab[..., 0] / 255.0).astype(np.float32)
    out = lab.copy()

    def gf(guide, src):
        d = radius
        mean_I = cv2.boxFilter(guide, -1, (d, d))
        mean_p = cv2.boxFilter(src, -1, (d, d))
        corr_I = cv2.boxFilter(guide * guide, -1, (d, d))
        corr_Ip = cv2.boxFilter(guide * src, -1, (d, d))
        var_I = corr_I - mean_I * mean_I
        cov_Ip = corr_Ip - mean_I * mean_p
        aa = cov_Ip / (var_I + eps)
        bb = mean_p - aa * mean_I
        return cv2.boxFilter(aa, -1, (d, d)) * guide + cv2.boxFilter(bb, -1, (d, d))

    out[..., 1] = gf(L, lab[..., 1] / 255.0) * 255.0
    out[..., 2] = gf(L, lab[..., 2] / 255.0) * 255.0
    return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


# ---------------------------------------------------------------- C: 掩膜
def body_cloth_mask(im: Image.Image) -> np.ndarray | None:
    """DWPose 关键点 → 躯干/衣物软掩膜（近似，无 SAM）。"""
    try:
        d, left_hand, right_hand = retouch._get_dwpose()
    except Exception:
        return None
    bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    persons = d.detect_full(bgr)
    if not persons:
        return None
    p = max(persons, key=lambda q: q["bbox"][4])
    k = p["kps"]
    mask = np.zeros((h, w), np.uint8)
    # COCO 身体点：5/6 肩，11/12 髋，13/14 膝，15/16 踝
    pts = [i for i in (5, 6, 11, 12, 13, 14) if k[i, 2] > 0.3]
    if len(pts) >= 3:
        cv2.fillConvexPoly(mask, cv2.convexHull(k[pts, :2].astype(np.int32)), 255)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    return mask.astype(np.float32) / 255.0


def mask_region_stabilize(im: Image.Image, mask: np.ndarray) -> Image.Image:
    """掩膜内：色相异常像素用区域中值色度拉回（稳定衣物/手臂区域着色）。"""
    lab = _lab(im)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    m = mask > 0.3
    if m.sum() < 100:
        return im
    ma, mb = float(np.median(a[m])), float(np.median(b[m]))
    out = lab.copy()
    out[..., 1] = lab[..., 1] * (1 - mask) + (ma + 128) * mask
    out[..., 2] = lab[..., 2] * (1 - mask) + (mb + 128) * mask
    return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))


# ---------------------------------------------------------------- 拼图
def _font(size):
    try:
        return ImageFont.truetype(FONT_BOLD, size)
    except Exception:
        return ImageFont.load_default()


def zoom_crop(im: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    x1, y1, x2, y2 = box
    w, h = im.size
    return im.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))


def compose(before, after, title, out_path, box):
    H = 720
    zb = zoom_crop(before, box).resize((int(zoom_crop(before, box).width * H / zoom_crop(before, box).height), H))
    za = zoom_crop(after, box).resize((int(zoom_crop(after, box).width * H / zoom_crop(after, box).height), H))
    bw = int(before.width * H / before.height)
    aa = int(after.width * H / after.height)
    gap, m, top, bottom = 20, 20, 56, 40
    canvas = Image.new("RGB", (m * 2 + bw + aa + gap * 2 + zb.width + za.width, top + H + bottom), (245, 246, 248))
    d = ImageDraw.Draw(canvas)
    x = m
    for im, label in [(before, "BEFORE"), (after, "AFTER")]:
        canvas.paste(im.resize((int(im.width * H / im.height), H)), (x, top))
        d.text((x, 20), label, font=_font(26), fill=(30, 30, 30))
        x += im.resize((int(im.width * H / im.height), H)).width + gap
    for im, label in [(zb, "手臂/衣物放大·BEFORE"), (za, "放大·AFTER")]:
        canvas.paste(im, (x, top))
        d.text((x, 20), label, font=_font(26), fill=(180, 30, 30) if "BEFORE" in label else (0, 110, 60))
        x += im.width + gap
    d.text((m, top + H + 8), title, font=_font(22), fill=(90, 90, 90))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


# ---------------------------------------------------------------- main
def prepare(im: Image.Image) -> Image.Image:
    im, _ = retouch.preprocess(im, "colorize")
    up = retouch.TiledRealESRGAN("realesrgan", tile=256)
    arr = up.upscale(np.asarray(im))
    u = Image.fromarray(arr)
    if max(u.size) > 2048:
        s = 2048 / max(u.size)
        u = u.resize((int(u.width * s), int(u.height * s)), Image.LANCZOS)
    return retouch.fit_for_work(u, 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--zoom", default="0.45,0.45,0.85,0.95", help="手臂/衣物放大框 x1,y1,x2,y2 比例")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    box = tuple(float(v) for v in args.zoom.split(","))

    orig = retouch.load_rgb(args.input)
    t0 = time.time()
    work = prepare(orig)
    print(f"[prep] work={work.size} in {time.time()-t0:.1f}s", flush=True)

    pipe = retouch.load_sdxl_img2img()
    base_prompt = retouch.PROMPTS["colorize"]["prompt"]
    base_neg = retouch.PROMPTS["colorize"]["negative"]

    cache: dict = {}

    def sdxl(strength, cfg):
        key = (round(strength, 3), round(cfg, 2))
        if key not in cache:
            t = time.time()
            outs, _ = retouch.run_diffusion(pipe, [work], "colorize", strength,
                                            args.steps, args.seed, cfg, base_prompt, base_neg)
            cache[key] = (outs[0], round(time.time() - t, 1))
        return cache[key]

    def _skin_and_cloth(c):
        base = retouch.chroma_transfer(c, work, 1.0)
        base, _ = retouch.fix_skin_color_cast(base)
        return mask_region_stabilize(base, mask)

    mask = None
    rows = []

    def emit(name, route, color_img, strength, cfg, post, dt, note=""):
        after = post(color_img)
        d = outdir / name
        d.mkdir(exist_ok=True)
        after.save(d / "after.png")
        met = metrics(work, after)
        met.update({"variant": name, "route": route, "strength": strength, "cfg": cfg,
                    "runtime_s": dt, "note": note})
        rows.append(met)
        compose(work, after, f"{name} | {route} | strength={strength} cfg={cfg} | "
                f"align={met['edge_align']} bleed={met['bleed_ratio']} Lshift={met['L_shift']} | {dt}s",
                d / "compare.png", box)
        (d / "meta.json").write_text(json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {name} {met}", flush=True)

    # ---- A: SDXL 调参 ----
    for name, s, cfg in [("A0_baseline_s080_cfg7", 0.80, 7.0),
                         ("A1_s045_cfg7", 0.45, 7.0),
                         ("A2_s035_cfg7", 0.35, 7.0),
                         ("A3_s045_cfg5", 0.45, 5.0)]:
        color, dt = sdxl(s, cfg)
        emit(name, "A-SDXL调参", color, s, cfg,
             lambda c: retouch.chroma_transfer(c, work, 1.0), dt)

    # ---- B: 色度层配准（在基线强度上）----
    color80, dt80 = sdxl(0.80, 7.0)
    emit("B1_flow_s080", "B-配准", color80, 0.80, 7.0,
         lambda c: retouch.chroma_transfer(register_color(c, work), work, 1.0), dt80, "光流配准")
    emit("B2_flow_guided_s080", "B-配准", color80, 0.80, 7.0,
         lambda c: guided_chroma(retouch.chroma_transfer(register_color(c, work), work, 1.0)), dt80,
         "光流配准+引导滤波")
    color45, dt45 = sdxl(0.45, 7.0)
    emit("B3_flow_s045", "B-配准", color45, 0.45, 7.0,
         lambda c: retouch.chroma_transfer(register_color(c, work), work, 1.0), dt45, "光流配准@0.45")

    # ---- C: 掩膜约束 ----
    mask = body_cloth_mask(work)
    if mask is not None:
        emit("C1_mask_s080", "C-掩膜", color80, 0.80, 7.0,
             lambda c: mask_region_stabilize(retouch.chroma_transfer(c, work, 1.0), mask), dt80,
             "躯干/衣物掩膜区域色度稳定")
        emit("C2_mask_skin_s080", "C-掩膜", color80, 0.80, 7.0,
             lambda c: _skin_and_cloth(c), dt80, "DWPose 脸/手/前臂 + 衣物掩膜")
    else:
        print("[C] DWPose 不可用，跳过 C 路", flush=True)

    (outdir / "summary.json").write_text(
        json.dumps({"input": args.input, "work_size": list(work.size), "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
