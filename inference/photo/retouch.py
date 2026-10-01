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


def preprocess(im: Image.Image, mode: str) -> tuple[Image.Image, dict]:
    steps: list[str] = []
    im, crop_info = auto_crop(im)
    if crop_info.get("cropped"):
        steps.append("auto_crop")
    if mode in ("repair", "colorize"):
        im = ImageOps.autocontrast(im, cutoff=0.5)
        steps.append("autocontrast")
    elif mode == "product":
        im = white_point(im)
        steps.append("white_point")
    return im, {"preprocess_steps": steps, **crop_info}


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
             "case_seed": args.seed} for p in files]


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
             "seed": c["case_seed"], "status": "pending", "timings": {}, "sizes": {}, "outputs": {}}
        t0 = time.time()
        try:
            orig = load_rgb(c["input"])
            m["sizes"]["before"] = list(orig.size)
            im, pre_info = preprocess(orig, c["mode"])
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
            if m["mode"] == "product":
                after = clean_white_bg(after)
            elif m["mode"] == "colorize" and work is not None:
                # AI 上色结果只取色度，亮度沿用修复后原图（保脸/保结构）
                after = chroma_transfer(after, work, strength=1.0)
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
