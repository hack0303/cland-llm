#!/usr/bin/env python3
"""AI 修图产线 v0 — 批量照片修复/超分/上色/商品图。

管线（每张）：① Pillow 预处理（自动裁剪 + 色阶/白点）→ ② RealESRGAN 4x 超分（分块）
→ ③ SDXL img2img 低强度修复/上色（提示词按 mode）→ ④ before/after 对比图 + meta.json

用法::

    # 目录批量（同 mode）
    python3 inference/photo/retouch.py --input DIR --output DIR --mode repair \
        --strength 0.25 --seed 42 --batch 1

    # 混合模式（cases.json manifest，见 docs/photo-retouch-v0.md）
    python3 inference/photo/retouch.py --manifest cases.json --output DIR

参数：
    --mode repair|upscale|colorize|product   （manifest 模式下为默认值）
    --strength   img2img 去噪强度（<=0.6；未给则用 mode 预设：repair .25 / colorize .45 / product .35）
    --seed       随机种子（复现用）
    --batch      同一模式下每次送 SDXL 的图片数（显存受限，默认 1）
    --work-res   SDXL 工作分辨率长边（默认 1024，P40 甜点）
    --max-side   ESRGAN 输出长边上限（默认 2048）
    --steps      SDXL 总步数（实际步数 = int(steps*strength)，默认 30）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from upscale import TiledRealESRGAN  # noqa: E402

SDXL_FP16_DIR = "/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0-fp16"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

PROMPTS = {
    "repair": {
        "prompt": ("restore this old damaged photograph, remove scratches noise stains and compression artifacts, "
                   "keep the original subject composition and details unchanged, natural realistic colors, "
                   "sharp clean high quality photo"),
        "negative": ("oversaturated, cartoon, illustration, painting, distorted, deformed, watermark, text, signature, "
                     "blurry, low quality, jpeg artifacts, extra objects"),
        "strength": 0.25,
        "guidance": 4.5,
    },
    "colorize": {
        "prompt": ("colorized historical photograph, natural true-to-life colors, accurate skin tones, "
                   "color photo, sharp details, high quality"),
        "negative": ("black and white, grayscale, greyscale, monochrome, sepia, oversaturated, neon, cartoon, "
                     "illustration, painting, distorted, deformed, watermark, text, blurry, low quality, extra objects"),
        "strength": 0.80,
        "guidance": 7.0,
    },
    "product": {
        "prompt": ("professional e-commerce product photograph, clean pure white background, studio softbox lighting, "
                   "accurate colors, sharp focus, centered product, high quality"),
        "negative": ("cluttered background, colorful background, shadow on background, text, watermark, logo, "
                     "distorted, deformed, extra objects, blurry, low quality"),
        "strength": 0.25,
        "guidance": 4.5,
    },
}


# --------------------------------------------------------------------- Pillow
def load_rgb(path: str | Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def auto_crop(im: Image.Image, max_frac: float = 0.12, std_thresh: float = 4.0) -> tuple[Image.Image, dict]:
    """检测并裁掉四周的纯色扫描/相纸边框（保守：单边最多 12%）。"""
    arr = np.asarray(im.convert("L"), np.float32)
    h, w = arr.shape
    rows = arr.std(axis=1)
    cols = arr.std(axis=0)

    def trim(vals: np.ndarray) -> tuple[int, int]:
        n = len(vals)
        lim = max(1, int(n * max_frac))
        a = 0
        while a < lim and vals[a] < std_thresh:
            a += 1
        b = 0
        while b < lim and vals[n - 1 - b] < std_thresh:
            b += 1
        return a, b

    ta, tb = trim(rows)
    la, lb = trim(cols)
    if (ta + tb + la + lb) == 0:
        return im, {"cropped": False}
    box = (la, ta, w - lb, h - tb)
    if box[2] - box[0] < w * 0.5 or box[3] - box[1] < h * 0.5:
        return im, {"cropped": False}
    out = im.crop(box)
    return out, {"cropped": True, "box": list(box), "removed_px": [la, ta, lb, tb]}


def white_point(im: Image.Image, border: float = 0.05) -> Image.Image:
    """商品图/扫描件白点归一：按边缘背景亮度把每通道拉到 255（只提亮，限幅 1.35x）。"""
    arr = np.asarray(im, np.float32)
    h, w = arr.shape[:2]
    b = max(1, int(min(h, w) * border))
    edge = np.concatenate([
        arr[:b].reshape(-1, 3), arr[-b:].reshape(-1, 3),
        arr[:, :b].reshape(-1, 3), arr[:, -b:].reshape(-1, 3),
    ])
    wp = np.percentile(edge, 90, axis=0)
    scale = np.clip(255.0 / np.maximum(wp, 1.0), 1.0, 1.35)
    return Image.fromarray(np.clip(arr * scale, 0, 255).astype(np.uint8))


def remove_damage(im: Image.Image, k: int = 9, thresh: int = 32,
                  max_comp_frac: float = 0.0015, dilate: int = 2) -> tuple[Image.Image, dict]:
    """老旧照片微小损伤去除（去划痕/去霉点/去白斑）——真实产线预处理步骤。

    思路：top-hat / black-hat 形态学检测**细小的亮/暗结构**（划痕、白斑、霉点），
    按连通域面积与长宽比过滤掉“内容”（人脸、线条等大面积或团块结构，避免涂抹失真），
    只对“薄小/细长”损伤做 Telea inpaint；随后仍走 SDXL 低强度修复以自然融合。
    返回 (处理图, 统计)。"""
    import cv2
    arr = np.asarray(im.convert("RGB"))
    g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    top = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kk)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kk)
    m = (((top > thresh) | (bh > thresh)).astype(np.uint8)) * 255
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    h, w = m.shape
    lim = max(4, max_comp_frac * h * w)
    keep = np.zeros_like(m)
    n_big = n_thin = 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        cw, ch = int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = max(cw, ch) / max(1, min(cw, ch))
        if area <= lim:
            keep[lab == i] = 255
            n_big += 1
        elif aspect >= 4 and area <= lim * 10:   # 细长划痕（可能被连成一条）
            keep[lab == i] = 255
            n_thin += 1
    keep = cv2.dilate(keep, np.ones((dilate, dilate), np.uint8), 1)
    if int(keep.sum()) == 0:
        return im, {"damage_fix": True, "damage_mask_pct": 0.0, "specks": 0, "scratches": 0}
    fixed = cv2.inpaint(arr, keep, 3, cv2.INPAINT_TELEA)
    return Image.fromarray(fixed), {"damage_fix": True,
                                   "damage_mask_pct": round(float((keep > 0).mean() * 100), 3),
                                   "specks": n_big, "scratches": n_thin}


def chroma_transfer(color_img: Image.Image, base_img: Image.Image, strength: float = 1.0) -> Image.Image:
    """LAB 色度迁移：把 color_img 的色度（a/b）搬到 base_img 的亮度（L）上。
    上色场景保结构：亮度/层次完全沿用修复后的原图，只采纳 AI 猜测的颜色。"""
    import cv2
    c = cv2.cvtColor(np.asarray(color_img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(np.asarray(base_img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    if c.shape[:2] != b.shape[:2]:
        c = cv2.resize(c, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    out = b.copy()
    out[..., 1] = np.clip(128 + (c[..., 1] - 128) * strength, 0, 255)
    out[..., 2] = np.clip(128 + (c[..., 2] - 128) * strength, 0, 255)
    return Image.fromarray(cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_LAB2RGB))


# ---- 肤色色偏修复（colorize 后处理）参数（针对褪色黑白老照片上色的脸/手臂伪影） ----
SKIN_HUE_RANGE = (22.0, 72.0)   # 判定“天然肤色”的 LAB 色相窗口（度）
SKIN_TARGET_HUE = 43.0          # 色偏像素强制拉回的暖肤色色相
SKIN_SAT_RANGE = (5.0, 10.0)    # 肤色 LAB 色度（饱和度）夹取区间

_dwpose_cache: tuple | None = None


def _get_dwpose():
    """惰性加载 DWPose（CPU onnxruntime）；不可用时返回 None。"""
    global _dwpose_cache
    if _dwpose_cache is None:
        hp = Path(__file__).resolve().parent.parent / "sdxl" / "hand_pipe"
        sys.path.insert(0, str(hp))
        from dwpose import DWPose, LEFT_HAND, RIGHT_HAND  # noqa: E402
        _dwpose_cache = (DWPose(), LEFT_HAND, RIGHT_HAND)
    return _dwpose_cache


def build_skin_mask(im: Image.Image) -> tuple[np.ndarray | None, list[int] | None]:
    """基于 DWPose 关键点（脸/手/前臂）生成软肤色掩膜（0..1）；不可用返回 (None, None)。
    眼睛区域挖洞，避免眼白/虹膜被当成肤色染色。第二个返回值为人脸 bbox [x1,y1,x2,y2]（供证据图裁剪）。"""
    import cv2
    try:
        d, left_hand, right_hand = _get_dwpose()
    except Exception as e:  # noqa: BLE001
        print(f"[skin] DWPose 不可用，跳过肤色修复: {e}", flush=True)
        return None, None
    rgb = np.asarray(im.convert("RGB"))
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        persons = d.detect_full(bgr)
    except Exception as e:  # noqa: BLE001
        print(f"[skin] DWPose 推理失败: {e}", flush=True)
        return None, None
    if not persons:
        return None, None
    p = max(persons, key=lambda q: q["bbox"][4])
    k = p["kps"]
    face_pts = [(int(x), int(y)) for x, y, c in k[23:91] if c > 0.3]
    face_box = ([min(p[0] for p in face_pts), min(p[1] for p in face_pts),
                 max(p[0] for p in face_pts), max(p[1] for p in face_pts)] if face_pts else None)
    mask = np.zeros((h, w), np.uint8)

    def poly(pts, conf: float = 0.3):
        pts = [(int(x), int(y)) for x, y, c in pts if c > conf]
        if len(pts) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(np.array(pts, np.int32)), 255)

    def limb(p1, p2, r1: int, r2: int):
        p1 = np.asarray(p1, float)
        p2 = np.asarray(p2, float)
        n = max(2, int(np.linalg.norm(p2 - p1)) // 4)
        for t in np.linspace(0, 1, n):
            cv2.circle(mask, tuple(np.round(p1 + (p2 - p1) * t).astype(int)),
                       int(r1 * (1 - t) + r2 * t), 255, -1)

    poly(k[23:91])                      # 脸（COCO-WholeBody face 68 点）
    poly(k[left_hand])                  # 左手
    poly(k[right_hand])                 # 右手
    for e, wr in ((7, 9), (8, 10)):     # 前臂（肘→腕，裸露段）
        if k[e, 2] > 0.3 and k[wr, 2] > 0.3:
            limb(k[e, :2], k[wr, :2], 30, 24)
    if k[5, 2] > 0.3 and k[6, 2] > 0.3:  # 颈/前胸（下巴→双肩中点）
        limb(k[23:91][:, :2].mean(0), (k[5, :2] + k[6, :2]) / 2, 34, 30)
    eye = np.zeros((h, w), np.uint8)
    for base in (36, 42):               # 68 点眼周 36-41/42-47 → +23
        pts = k[23 + base: 23 + base + 6, :2].astype(np.int32)
        cv2.fillConvexPoly(eye, cv2.convexHull(pts), 255)
    mask[cv2.dilate(eye, np.ones((7, 7), np.uint8), 1) > 0] = 0
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), 1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return mask.astype(np.float32) / 255.0, face_box


def fix_skin_color_cast(im: Image.Image, mask: np.ndarray | None = None) -> tuple[Image.Image, dict]:
    """抑制上色结果中脸/手臂区域的局部色偏伪影（蓝/青/洋红色块）。
    思路：在肤色掩膜内，把色相异常的像素拉回暖肤色（保持亮度/纹理），并夹取饱和度；掩膜外不动。"""
    import cv2
    face_box = None
    if mask is None:
        mask, face_box = build_skin_mask(im)
    if mask is None or float(mask.max()) < 0.05:
        return im, {"skin_fix": False}
    rgb = np.asarray(im.convert("RGB")).astype(np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    sat = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a))
    lo, hi = SKIN_HUE_RANGE
    warm = (hue >= lo) & (hue <= hi)
    hue2 = np.where(warm, hue, SKIN_TARGET_HUE)
    sat2 = np.clip(sat, *SKIN_SAT_RANGE)
    a2 = sat2 * np.cos(np.radians(hue2))
    b2 = sat2 * np.sin(np.radians(hue2))
    m = mask
    lab[..., 1] = lab[..., 1] * (1 - m) + (a2 + 128) * m
    lab[..., 2] = lab[..., 2] * (1 - m) + (b2 + 128) * m
    out = Image.fromarray(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    skin_px = int((m > 0.5).sum())
    cast_pct = float(((~warm) & (m > 0.5)).sum()) / max(1, skin_px) * 100
    return out, {"skin_fix": True, "skin_area_pct": round(float((m > 0.5).mean() * 100), 2),
                 "skin_cast_fixed_pct": round(cast_pct, 2), "face_box": face_box}


def compose_skin_zoom(before: Image.Image, after: Image.Image, face_box: list[int] | None,
                      out_path: str | Path, panel_w: int = 460) -> bool:
    """生成「修复前/后」面部放大对照图（供验收佐证）。face_box=[x1,y1,x2,y2]（同一坐标系）。"""
    if not face_box:
        return False
    x1, y1, x2, y2 = face_box
    w, h = before.size
    fw, fh = x2 - x1, y2 - y1
    bx1, by1 = max(0, int(x1 - fw * 0.55)), max(0, int(y1 - fh * 0.55))
    bx2, by2 = min(w, int(x2 + fw * 0.55)), min(h, int(y2 + fh * 0.75))
    b, a = before.crop((bx1, by1, bx2, by2)), after.crop((bx1, by1, bx2, by2))

    def fit(im: Image.Image) -> Image.Image:
        s = panel_w / im.width
        return im.resize((panel_w, max(1, int(im.height * s))), Image.LANCZOS)

    b, a = fit(b), fit(a)
    gap, margin, top, bottom = 18, 20, 50, 14
    canvas = Image.new("RGB", (margin * 2 + b.width + gap + a.width, top + b.height + bottom), (246, 246, 246))
    d = ImageDraw.Draw(canvas)
    d.text((margin, 12), "修复前（AI 原上色 · 色偏）", font=_font(FONT_BOLD, 24), fill=(180, 30, 30))
    d.text((margin + b.width + gap, 12), "修复后（肤色色偏修复）", font=_font(FONT_BOLD, 24), fill=(0, 110, 60))
    canvas.paste(b, (margin, top))
    canvas.paste(a, (margin + b.width + gap, top))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return True


def clean_white_bg(im: Image.Image, thresh: int = 240) -> Image.Image:
    """商品图收尾：把与图像边缘连通的近白区域刷成纯白（保留被产品包围的高光）。"""
    import cv2
    arr = np.asarray(im, np.uint8)
    near_white = (arr.min(axis=2) >= thresh).astype(np.uint8)
    if near_white.sum() == 0:
        return im
    num, labels = cv2.connectedComponents(near_white, connectivity=4)
    edges = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    keep_labels = set(edges.tolist()) - {0}
    if not keep_labels:
        return im
    keep = np.isin(labels, list(keep_labels))
    out = arr.copy()
    out[keep] = 255
    return Image.fromarray(out)


def preprocess(im: Image.Image, mode: str, damage_fix: bool = False) -> tuple[Image.Image, dict]:
    steps: list[str] = []
    info: dict = {}
    im, crop_info = auto_crop(im)
    if crop_info.get("cropped"):
        steps.append("auto_crop")
    if mode in ("repair", "colorize"):
        im = ImageOps.autocontrast(im, cutoff=0.5)
        steps.append("autocontrast")
    elif mode == "product":
        im = white_point(im)
        steps.append("white_point")
    if mode == "repair" and damage_fix:
        im, dinfo = remove_damage(im)
        steps.append("remove_damage")
        info.update(dinfo)
    return im, {"preprocess_steps": steps, **info, **crop_info}


def fit_for_work(im: Image.Image, work_res: int) -> Image.Image:
    """把长边缩到 work_res（不放大），并取 16 的倍数（SDXL 要求）。"""
    w, h = im.size
    scale = work_res / max(w, h)
    if scale < 1.0:
        w, h = int(w * scale), int(h * scale)
    w = max(256, (w // 16) * 16)
    h = max(256, (h // 16) * 16)
    if (w, h) != im.size:
        im = im.resize((w, h), Image.LANCZOS)
    return im


# ---------------------------------------------------------------------- SDXL
def load_sdxl_img2img():
    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline
    t0 = time.time()
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        SDXL_FP16_DIR, torch_dtype=torch.float16, use_safetensors=True)
    pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    print(f"[sdxl] loaded in {time.time() - t0:.1f}s", flush=True)
    return pipe


def run_diffusion(pipe, images: list[Image.Image], mode: str, strength: float,
                  steps: int, seed: int, guidance: float,
                  prompt: str | None = None, negative: str | None = None) -> list[Image.Image]:
    import torch
    cfg = PROMPTS[mode]
    g = torch.Generator("cpu").manual_seed(seed)
    t0 = time.time()
    out = pipe(
        prompt=prompt or cfg["prompt"],
        negative_prompt=negative or cfg["negative"],
        image=images,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=g,
    ).images
    dt = time.time() - t0
    print(f"[sdxl] {len(images)} img × {mode} strength={strength} "
          f"steps={steps}(≈{max(1, int(steps * strength))}) {dt:.1f}s", flush=True)
    return out, round(dt, 2)


# -------------------------------------------------------------------- compare
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def compose_compare(before: Image.Image, after: Image.Image, case: str,
                    subtitle: str, out_path: str | Path, height: int = 540) -> None:
    def scaled(im: Image.Image) -> Image.Image:
        w, h = im.size
        s = height / h
        return im.resize((max(1, int(w * s)), height), Image.LANCZOS)

    b, a = scaled(before), scaled(after)
    gap, margin, top, bottom = 24, 24, 52, 46
    w = margin * 2 + b.width + gap + a.width
    h = top + height + bottom
    canvas = Image.new("RGB", (w, h), (246, 246, 246))
    d = ImageDraw.Draw(canvas)
    f_label = _font(FONT_BOLD, 26)
    f_sub = _font(FONT_REGULAR, 20)
    d.text((margin, 12), "BEFORE（输入）", font=f_label, fill=(30, 30, 30))
    d.text((margin + b.width + gap, 12), "AFTER（输出）", font=f_label, fill=(0, 110, 60))
    canvas.paste(b, (margin, top))
    canvas.paste(a, (margin + b.width + gap, top))
    d.text((margin, top + height + 10), subtitle, font=f_sub, fill=(90, 90, 90))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def sha256_file(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------- main
def load_cases(args) -> list[dict]:
    if args.manifest:
        raw = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        cases = []
        for i, c in enumerate(raw):
            cases.append({
                "name": c.get("name") or f"case{i + 1:02d}",
                "input": c["input"],
                "mode": c.get("mode", args.mode),
                "strength": c.get("strength", args.strength),
                "steps": c.get("steps", args.steps),
                "guidance": c.get("guidance", args.guidance),
                "gt": c.get("gt"),
                "note": c.get("note", ""),
                "prompt": c.get("prompt"),
                "negative": c.get("negative"),
                "skin_fix": c.get("skin_fix"),
                "damage_fix": c.get("damage_fix", args.damage_fix),
                "case_seed": c.get("seed", args.seed),
            })
        return cases
    in_dir = Path(args.input)
    if in_dir.is_file():
        files = [in_dir]
    else:
        files = sorted(p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXT)
    return [{"name": p.stem, "input": str(p), "mode": args.mode,
             "strength": args.strength, "steps": args.steps, "guidance": args.guidance,
             "gt": None, "note": "", "prompt": None, "negative": None,
             "skin_fix": None, "damage_fix": args.damage_fix, "case_seed": args.seed} for p in files]


def main():
    ap = argparse.ArgumentParser(description="AI 修图产线 v0")
    ap.add_argument("--input", help="输入目录或单文件（无 manifest 时使用）")
    ap.add_argument("--manifest", help="cases.json：逐案例 input/mode/strength/gt")
    ap.add_argument("--output", required=True, help="输出目录（产出 <case>/{before,after,compare,meta}）")
    ap.add_argument("--mode", default="repair", choices=["repair", "upscale", "colorize", "product"])
    ap.add_argument("--strength", type=float, default=None, help="img2img 去噪强度（默认按 mode）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1, help="同 mode 下每次送 SDXL 的图片数")
    ap.add_argument("--work-res", type=int, default=1024, help="SDXL 工作分辨率长边")
    ap.add_argument("--max-side", type=int, default=2048, help="ESRGAN 输出长边上限")
    ap.add_argument("--steps", type=int, default=80, help="SDXL 总步数（实际步数 = int(steps*strength)）")
    ap.add_argument("--guidance", type=float, default=None, help="CFG（默认按 mode：colorize 7.0 其余 4.5）")
    ap.add_argument("--skin-fix", action="store_true",
                    help="colorize 后处理启用脸/手臂肤色色偏修复（默认关；也可在 manifest 逐案例 skin_fix）")
    ap.add_argument("--damage-fix", action="store_true",
                    help="repair 预处理启用去划痕/去白斑/去霉点（形态学检测 + inpaint；也可在 manifest 逐案例 damage_fix）")
    ap.add_argument("--esrgan", default="realesrgan", choices=["realesrgan", "ultrasharp"])
    ap.add_argument("--esrgan-in-cap", type=int, default=1280,
                    help="ESRGAN 输入长边上限（x4 输出数组内存保护，默认 1280）")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--run-name", default=None, help="run 标识（默认时间戳）")
    args = ap.parse_args()

    if not args.manifest and not args.input:
        ap.error("需要 --input 或 --manifest 之一")
    run_name = args.run_name or time.strftime("%Y%m%d-%H%M%S")
    out_root = Path(args.output)
    cases = load_cases(args)
    if not cases:
        print("[!] 没有可处理的图片")
        return 1

    for c in cases:
        if c["strength"] is None:
            c["strength"] = PROMPTS.get(c["mode"], {}).get("strength", 0.3)
        if c.get("guidance") is None:
            c["guidance"] = PROMPTS.get(c["mode"], {}).get("guidance", 4.5)

    # ---------- Phase A: 预处理 + 超分 ----------
    up: TiledRealESRGAN | None = None
    metas: list[dict] = []
    for c in cases:
        m = {"case": c["name"], "mode": c["mode"], "input": str(c["input"]),
             "gt": c.get("gt"), "note": c.get("note", ""), "strength": c["strength"],
             "steps": c["steps"], "guidance": c["guidance"],
             "prompt": c.get("prompt"), "negative": c.get("negative"),
             "skin_fix": c.get("skin_fix"), "damage_fix": c.get("damage_fix", False),
             "seed": c["case_seed"], "status": "pending", "timings": {}, "sizes": {}, "outputs": {}}
        t0 = time.time()
        try:
            orig = load_rgb(c["input"])
            m["sizes"]["before"] = list(orig.size)
            im, pre_info = preprocess(orig, c["mode"], damage_fix=bool(c.get("damage_fix")))
            m.update(pre_info)
            t_pre = time.time() - t0

            t1 = time.time()
            if up is None:
                up = TiledRealESRGAN(args.esrgan, tile=args.tile)
            # 大图先规整到 esrgan-in-cap，避免 4x 输出数组过大（1280 -> 5120）
            if max(im.size) > args.esrgan_in_cap:
                s = args.esrgan_in_cap / max(im.size)
                im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
                m["esrgan_in_resized"] = list(im.size)
            out_arr = up.upscale(np.asarray(im))
            up_img = Image.fromarray(out_arr)
            if max(up_img.size) > args.max_side:
                s = args.max_side / max(up_img.size)
                up_img = up_img.resize((int(up_img.width * s), int(up_img.height * s)), Image.LANCZOS)
            t_up = time.time() - t1

            work = fit_for_work(up_img, args.work_res)
            m["timings"] = {"preprocess": round(t_pre, 2), "esrgan": round(t_up, 2)}
            m["sizes"]["upscaled"] = list(up_img.size)
            m["sizes"]["work"] = list(work.size)
            m["_work"] = work
            m["_up"] = up_img
            m["_orig"] = orig
            m["status"] = "prepared"
            print(f"[prep] {c['name']}: {m['sizes']['before']} -> up {m['sizes']['upscaled']} -> work {work.size} "
                  f"({t_pre + t_up:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            m["status"] = "failed"
            m["error"] = f"{type(e).__name__}: {e}"
            print(f"[FAIL] {c['name']}: {m['error']}", flush=True)
        metas.append(m)

    # ---------- Phase B: SDXL 修复/上色（按 mode 分批） ----------
    pipe = None
    prepared = [m for m in metas if m["status"] == "prepared" and m["mode"] != "upscale"]
    idx = 0
    while idx < len(prepared):
        grp = [prepared[idx]]
        j = idx + 1
        while (j < len(prepared) and len(grp) < args.batch
               and prepared[j]["mode"] == prepared[idx]["mode"]
               and prepared[j]["strength"] == prepared[idx]["strength"]
               and prepared[j]["steps"] == prepared[idx]["steps"]
               and prepared[j]["prompt"] == prepared[idx]["prompt"]
               and prepared[j]["negative"] == prepared[idx]["negative"]
               and prepared[j]["seed"] == prepared[idx]["seed"]):
            grp.append(prepared[j])
            j += 1
        idx = j
        try:
            if pipe is None:
                pipe = load_sdxl_img2img()
            t0 = time.time()
            outs, dt = run_diffusion(pipe, [m["_work"] for m in grp], grp[0]["mode"],
                                     grp[0]["strength"], grp[0]["steps"], grp[0]["seed"],
                                     grp[0]["guidance"], grp[0].get("prompt"), grp[0].get("negative"))
            for m, o in zip(grp, outs):
                m["_out"] = o
                m["timings"]["diffusion"] = round(dt / len(grp), 2)
                m["status"] = "done"
        except Exception as e:  # noqa: BLE001
            for m in grp:
                m["status"] = "failed"
                m["error"] = f"diffusion: {type(e).__name__}: {e}"
                print(f"[FAIL] diffusion {m['case']}: {m['error']}", flush=True)
    for m in metas:  # upscale 模式无需扩散，直接落盘（输出 ESRGAN 封顶图，非工作分辨率）
        if m["status"] == "prepared":
            m["_out"] = m["_up"] if m["mode"] == "upscale" else m["_work"]
            m["status"] = "done"

    # ---------- Phase C: 收尾（商品白底）+ 保存 + 对比图 ----------
    for m in metas:
        if m["status"] != "done":
            continue
        try:
            t0 = time.time()
            after = m.pop("_out")
            before = m.pop("_orig")
            work = m.get("_work")
            skin_zoom_src = None
            if m["mode"] == "product":
                after = clean_white_bg(after)
            elif m["mode"] == "colorize" and work is not None:
                # AI 上色结果只取色度，亮度沿用修复后原图（保脸/保结构）
                after = chroma_transfer(after, work, strength=1.0)
                # 脸/手臂肤色色偏伪影修复（面部掩膜 + 局部色偏抑制）
                enable_skin = m.get("skin_fix")
                if enable_skin is None:
                    enable_skin = args.skin_fix
                if enable_skin:
                    skin_zoom_src = after.copy()
                    after, skin_info = fix_skin_color_cast(after)
                    m.update(skin_info)
            elif m["mode"] == "repair":
                # 黑白老照片保持黑白（修复不加色；上色请用 colorize 模式）
                barr = np.asarray(before, np.float32)
                if float(np.abs(barr[..., 0] - barr[..., 1]).mean()
                         + np.abs(barr[..., 2] - barr[..., 1]).mean()) < 2.0:
                    after = after.convert("L").convert("RGB")
                    m["grayscale_input"] = True
            case_dir = out_root / m["case"]
            case_dir.mkdir(parents=True, exist_ok=True)
            m.pop("_work", None)
            m.pop("_up", None)
            before.save(case_dir / "before.png")
            after.save(case_dir / "after.png")
            m["sizes"]["after"] = list(after.size)
            m["timings"]["post"] = round(time.time() - t0, 2)
            m["timings"]["total"] = round(sum(m["timings"].values()), 2)
            subtitle = (f"mode={m['mode']}  strength={m['strength']}  seed={m['seed']}  "
                        f"steps={m['steps']}  {m['sizes']['before']} → {m['sizes']['after']}  "
                        f"{m['timings']['total']}s")
            compose_compare(before, after, m["case"], subtitle, case_dir / "compare.png")
            m["outputs"] = {"before": str(case_dir / "before.png"),
                            "after": str(case_dir / "after.png"),
                            "compare": str(case_dir / "compare.png"),
                            "sha256_after": sha256_file(case_dir / "after.png")}
            if skin_zoom_src is not None:
                zoom_path = case_dir / "skin_zoom.png"
                if compose_skin_zoom(skin_zoom_src, after, m.get("face_box"), zoom_path):
                    m["outputs"]["skin_zoom"] = str(zoom_path)
            (case_dir / "meta.json").write_text(json.dumps(m, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
            print(f"[done] {m['case']}: {m['timings']['total']}s -> {case_dir}/compare.png", flush=True)
        except Exception as e:  # noqa: BLE001
            m["status"] = "failed"
            m["error"] = f"post: {type(e).__name__}: {e}"
            print(f"[FAIL] post {m['case']}: {m['error']}", flush=True)

    # ---------- run.json ----------
    ok = [m for m in metas if m["status"] == "done"]
    failed = [m for m in metas if m["status"] != "done"]
    total_s = sum(m["timings"].get("total", 0) for m in ok)
    # per-stage 均值
    stage_avg: dict[str, float] = {}
    for stage in ("preprocess", "esrgan", "diffusion", "post", "total"):
        vals = [m["timings"][stage] for m in ok if stage in m["timings"]]
        stage_avg[stage] = round(sum(vals) / len(vals), 2) if vals else 0.0
    summary = {
        "cases": len(metas), "ok": len(ok), "failed": len(failed),
        "fail_rate": round(len(failed) / len(metas), 4) if metas else 0,
        "total_seconds": round(total_s, 1),
        "stage_avg_seconds": stage_avg,
    }
    for m in metas:  # 清掉内存态对象，保证可 JSON 序列化
        m.pop("_orig", None)
        m.pop("_work", None)
        m.pop("_up", None)
        m.pop("_out", None)
    run = {"run": run_name, "args": {k: str(v) for k, v in vars(args).items()},
           "summary": summary, "cases": metas}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, ensure_ascii=False))
    print(f"[run] -> {out_root}/run.json")
    return 1 if summary["fail_rate"] > 0.05 else 0


if __name__ == "__main__":
    sys.exit(main())
