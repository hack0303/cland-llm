#!/usr/bin/env python3
"""#226 老照片上色模型评测基准：指标口径 + 基线对比。

配合 base/cland-crawler#226（DDColor 原理/技术调研），由 curie（评测）定义口径并出基线数据。

两条评测轨（详见 docs/ddcolor-评测-指标口径与基线对比.md）：

  Track A · 参考基准（有 GT，合成灰度上色）
    取彩色 GT -> 只保留亮度（灰度）-> 各模型上色 -> 与 GT 逐指标对比。
    指标：PSNR / SSIM / LPIPS / ΔE00（全局色差）/ BW-ΔE00（边界加权色差，抓"串色/错位"）
          / MI（错位指数 = ΔE_raw/ΔE_warp，>1 说明误差更多来自位置偏移而非语义错色）
          / colorfulness（自然度代理）

  Track B · 无参考（真实黑白老照片，无 GT）
    指标沿用 #225：skin_blue%（皮肤掩膜内蓝青色占比，主指标）/ whole_blue%
          / edge_align（色度梯度-亮度梯度相关，越高越贴结构）/ bleed_ratio（平滑区串色）
          / colorfulness / L_shift（相对输入亮度偏移）

基线变体：
  gray              —— 灰度直通（下界，colorfulness=0）
  ddcolor_native    —— DDColor-L 原生输出
  ddcolor_transfer  —— DDColor 出色 + chroma_transfer 保 work 亮度（产线 D2 口径）
  sdxl_s045_cfg7    —— SDXL img2img colorize strength=0.45
  sdxl_s060_cfg7    —— SDXL img2img colorize strength=0.60
  sdxl_s080_cfg7    —— SDXL img2img colorize strength=0.80（#225 原基线）
  deoldify          —— 预留（需 DEOLDIFY_REPO；未接入时自动跳过）

用法：
  python3 inference/photo/tools/bench_colorize.py \
      --track-a-manifest materials/bench-a.json \
      --track-b-dir "/mnt/data/ai_workspace/DDColor/assets/test_images" \
      --out /mnt/data/ai_workspace/outputs/colorize-bench-226 --work-res 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parent))
import retouch  # noqa: E402

# ------------------------------------------------------------------ 基础工具
def load_rgb(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def to_gray_rgb(im: Image.Image) -> Image.Image:
    """只保留亮度：RGB->灰度->RGB（模拟黑白输入，且 L 与 GT 完全一致）。"""
    return im.convert("L").convert("RGB")


def _arr(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("RGB"), np.float32)


def _lab(im: Image.Image) -> np.ndarray:
    """skimage LAB（L 0-100, a/b ~ -128..127）。"""
    return rgb2lab(_arr(im) / 255.0)


def _grad(m: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(m.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(m.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def colorfulness(im: Image.Image) -> float:
    """Hasler & Süsstrunk (2003) colorfulness。"""
    r, g, b = (_arr(im)[..., i] for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


# ------------------------------------------------------------------ Track A 指标
_lpips_fn = None


def _get_lpips():
    global _lpips_fn
    if _lpips_fn is None:
        import lpips
        import torch
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        _lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(_dev)
        _lpips_fn.eval()
    return _lpips_fn


def lpips_score(gt: Image.Image, pred: Image.Image) -> float:
    import torch
    fn = _get_lpips()
    size = (256, 256)
    dev = next(fn.parameters()).device

    def prep(im):
        a = np.asarray(im.convert("RGB").resize(size, Image.LANCZOS), np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)

    with torch.no_grad():
        return float(fn(prep(gt), prep(pred)).item())


def flow_warp(pred: Image.Image, ref: Image.Image) -> Image.Image:
    """用 ref 亮度为准，把 pred 通过稠密光流对齐到 ref 坐标系。"""
    ref_l = cv2.cvtColor(np.asarray(ref.convert("RGB")), cv2.COLOR_RGB2GRAY)
    pred_l = cv2.cvtColor(np.asarray(pred.convert("RGB")), cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(ref_l, pred_l, None, 0.5, 5, 25, 5, 7, 1.5, 0)
    h, w = ref_l.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    warped = cv2.remap(np.asarray(pred.convert("RGB")), xx + flow[..., 0], yy + flow[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(warped)


def _shift_image(im: Image.Image, dx: float, dy: float) -> Image.Image:
    a = np.asarray(im.convert("RGB"))
    h, w = a.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
    return Image.fromarray(cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REFLECT))


def chroma_misplacement(gt: Image.Image, pred: Image.Image) -> float:
    """错位指数 MI = ΔE00_raw / ΔE00_shift（全局色度平移补偿后）。

    用相位相关在 GT/pred 的 a、b 通道上估计最优全局平移；若颜色层整体平移即可
    大幅降低色差，说明误差主要来自「颜色覆盖位置偏移」而非语义错色。MI→1 表示
    无可补偿的全局错位；MI 越大错位越明显。位移不可靠（超图幅 15%）时回退 1。
    """
    gt_lab, pr_lab = _lab(gt), _lab(pred)
    dE_raw = float(deltaE_ciede2000(gt_lab, pr_lab).mean())
    h, w = gt_lab.shape[:2]
    shifts = []
    for ch in (1, 2):
        g = gt_lab[..., ch]
        p = pr_lab[..., ch]
        if g.std() < 1e-3 or p.std() < 1e-3:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(g.astype(np.float32), p.astype(np.float32))
        if abs(dx) <= 0.15 * w and abs(dy) <= 0.15 * h:
            shifts.append((dx, dy))
    if not shifts:
        return 1.0
    dx = float(np.mean([s[0] for s in shifts]))
    dy = float(np.mean([s[1] for s in shifts]))
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return 1.0
    dE_shift = float(deltaE_ciede2000(gt_lab, _lab(_shift_image(pred, dx, dy))).mean())
    if dE_shift <= 1e-6:
        return 1.0
    # 只能补偿（降低色差）：ratio<1 说明平移反而更差 -> 无有效全局错位
    return round(max(1.0, dE_raw / dE_shift), 3)


def track_a_metrics(gt: Image.Image, pred: Image.Image) -> dict:
    gt_a, pr_a = _arr(gt), _arr(pred)
    psnr_v = float(psnr(gt_a, pr_a, data_range=255))
    ssim_v = float(ssim(gt_a, pr_a, channel_axis=2, data_range=255))
    lp = lpips_score(gt, pred)
    gt_lab, pr_lab = _lab(gt), _lab(pred)
    dE = deltaE_ciede2000(gt_lab, pr_lab)
    # 边界加权色差：GT 亮度梯度 top10% 像素
    gL = _grad(gt_lab[..., 0])
    thr = np.percentile(gL, 90)
    edge = gL >= thr
    bw_de = float(dE[edge].mean()) if edge.sum() > 50 else float("nan")
    de_raw = float(dE.mean())
    mi = chroma_misplacement(gt, pred)
    return {
        "psnr": round(psnr_v, 2),
        "ssim": round(ssim_v, 4),
        "lpips": round(lp, 4),
        "dE00": round(de_raw, 2),
        "bw_dE00": round(bw_de, 2),
        "misplace_idx": round(mi, 3),
        "colorfulness": round(colorfulness(pred), 1),
    }


# ------------------------------------------------------------------ Track B 指标
def _skin_mask(im: Image.Image):
    try:
        return retouch.build_skin_mask(im)
    except Exception:
        return None, None


def _blue_pct(im: Image.Image, mask=None) -> float:
    """蓝青伪影占比——与 #225 `blue_artifact_pct` 同口径：Lab a/b → sat/hue，
    sat>18 且 hue∈(-140°,-40°) 记为蓝青（要求饱和度，避开暗部肤色误判）。"""
    lab = _lab(im)
    sat = np.hypot(lab[..., 1], lab[..., 2])
    hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    blue = (sat > 18) & (hue > -140) & (hue < -40)
    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.sum() < 50:
            return 0.0
        return float(blue[mask].mean() * 100)
    return float(blue.mean() * 100)


def track_b_metrics(src: Image.Image, pred: Image.Image) -> dict:
    mask, _ = _skin_mask(src)
    pred_lab = _lab(pred)
    ab = np.hypot(pred_lab[..., 1], pred_lab[..., 2])
    # 结构边缘来自输入 src 的 L（#225 口径：结构=L，色度=输出）
    gL = _grad(_lab(src)[..., 0])
    gC = _grad(ab)
    thr = np.percentile(gL, 90)
    edge = gL >= thr
    if edge.sum() > 50 and gC[edge].std() > 1e-6 and gL[edge].std() > 1e-6:
        align = float(np.corrcoef(gL[edge], gC[edge])[0, 1])
    else:
        align = 0.0
    smooth = gL <= np.percentile(gL, 40)
    bleed = float(gC[smooth].mean() / (gC.mean() + 1e-6))
    l_shift = float(np.mean(np.abs(
        _lab(src)[..., 0] - pred_lab[..., 0])))
    return {
        "skin_blue_pct": round(_blue_pct(pred, mask), 3),
        "whole_blue_pct": round(_blue_pct(pred), 3),
        "edge_align": round(align, 4),
        "bleed_ratio": round(bleed, 4),
        "colorfulness": round(colorfulness(pred), 1),
        "L_shift": round(l_shift, 3),
    }


# ------------------------------------------------------------------ 变体执行
def run_variant(name: str, src_rgb: Image.Image, sdxl_pipe, ddcolor_pipe):
    """src_rgb = 送模型的工作图（Track A 已是灰度、Track B 为原图）。返回上色输出（已 chroma_transfer 绑回 L）。"""
    if name == "gray":
        return src_rgb
    if name == "ddcolor_native":
        if ddcolor_pipe is None:
            ddcolor_pipe = retouch.load_ddcolor()
        out, _ = retouch.run_ddcolor(ddcolor_pipe, [src_rgb])
        return out[0]
    if name == "ddcolor_transfer":
        if ddcolor_pipe is None:
            ddcolor_pipe = retouch.load_ddcolor()
        out, _ = retouch.run_ddcolor(ddcolor_pipe, [src_rgb])
        return retouch.chroma_transfer(out[0], src_rgb, 1.0)
    if name.startswith("sdxl_"):
        _, strength, cfg = name.split("_")
        s = float(strength[1:]) / 100.0
        g = float(cfg[3:])
        if sdxl_pipe is None:
            sdxl_pipe = retouch.load_sdxl_img2img()
        out, _ = retouch.run_diffusion(sdxl_pipe, [src_rgb], "colorize", s, 80, 42, g)
        return retouch.chroma_transfer(out[0], src_rgb, 1.0)
    if name == "deoldify":
        return _run_deoldify(src_rgb)
    raise ValueError(f"unknown variant: {name}")


def _run_deoldify(src_rgb: Image.Image):
    """DeOldify 预留接口：需安装并设置 DEOLDIFY_REPO + DEOLDIFY_WEIGHTS。"""
    repo = os.environ.get("DEOLDIFY_REPO")
    if not repo or not Path(repo).exists():
        raise RuntimeError("DEOLDIFY_REPO 未设置/不存在，跳过 DeOldify 变体")
    sys.path.insert(0, repo)
    from deoldify import device  # noqa: F401
    from deoldify.visualize import get_image_colorizer
    colorizer = get_image_colorizer(artistic=True)
    return colorizer.plot_transformed_image_from_image(src_rgb, render_factor=35)


# ------------------------------------------------------------------ 统计
def _wilcoxon(a, b):
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


def _cliffs_delta(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    gt = sum((x > y) for x in a for y in b)
    lt = sum((x < y) for x in a for y in b)
    n = len(a) * len(b)
    return (gt - lt) / n if n else 0.0


def _boot_ci(a, b, n=10000, seed=20261002):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    n_obs = len(a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        diffs.append(np.median(a[idx] - b[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def significance_table(per_image: list[dict], metrics: list[str], baseline: str, variants: list[str]):
    """每个变体 vs baseline 的配对统计（Wilcoxon + bootstrap 中位差 CI + Cliff's delta）。"""
    rows = []
    for v in variants:
        if v == baseline:
            continue
        for m in metrics:
            a = [r["metrics"][m] for r in per_image if r["variant"] == v and not np.isnan(r["metrics"][m])]
            b = [r["metrics"][m] for r in per_image if r["variant"] == baseline and not np.isnan(r["metrics"][m])]
            n = min(len(a), len(b))
            a, b = a[:n], b[:n]
            if n == 0:
                continue
            lo, hi = _boot_ci(np.array(a), np.array(b))
            rows.append({
                "variant": v, "baseline": baseline, "metric": m, "n": n,
                "median_variant": round(float(np.median(a)), 3),
                "median_baseline": round(float(np.median(b)), 3),
                "wilcoxon_p": round(_wilcoxon(a, b), 4) if n >= 8 else "n<8",
                "boot_ci95_delta": [round(lo, 3), round(hi, 3)],
                "cliffs_delta": round(_cliffs_delta(a, b), 3),
            })
    return rows


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-a-manifest", help="JSON: [{name, gt}] 彩色 GT 列表（Track A）")
    ap.add_argument("--track-b-dir", help="目录：真实黑白老照片（Track B）")
    ap.add_argument("--track-b-list", help="JSON: 文件名列表（可选，过滤 track-b-dir）")
    ap.add_argument("--variants", default="ddcolor_transfer,ddcolor_native,sdxl_s045_cfg7,"
                                           "sdxl_s060_cfg7,sdxl_s080_cfg7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work-res", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="每轨最多取 N 张（0=全部）")
    ap.add_argument("--recompute", action="store_true",
                    help="不跑模型，从已保存 PNG 重算指标并重建汇总")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    variants = [v for v in args.variants.split(",") if v]

    if args.recompute:
        rows = recompute_from_disk(outdir, variants, args.work_res, args.track_b_dir)
        (outdir / "per_image.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        write_summaries(outdir, rows, variants)
        print(f"\n[recompute done] {len(rows)} rows -> {outdir}", flush=True)
        return

    sdxl_pipe = ddcolor_pipe = None
    per_image: list[dict] = []
    # 增量合并：同一 (track,image,variant) 的新结果覆盖旧行，避免分次跑互相覆盖
    prev = {}
    pj = outdir / "per_image.json"
    if pj.exists():
        for r in json.loads(pj.read_text(encoding="utf-8")):
            prev[(r["track"], r["image"], r["variant"])] = r

    def process(track, name, src: Image.Image):
        nonlocal sdxl_pipe, ddcolor_pipe
        src = retouch.fit_for_work(src, args.work_res)
        gt = src if track == "A" else None
        inp = to_gray_rgb(src) if track == "A" else src
        for v in variants:
            t0 = time.time()
            try:
                pred = run_variant(v, inp, sdxl_pipe, ddcolor_pipe)
            except RuntimeError as e:
                print(f"[skip] {v}: {e}", flush=True)
                continue
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {name} / {v}: {type(e).__name__}: {e}", flush=True)
                continue
            dt = round(time.time() - t0, 2)
            di = outdir / track / name
            di.mkdir(parents=True, exist_ok=True)
            pred.save(di / f"{v}.png")
            if track == "A":
                src.save(di / "input_gray.png")
                gt.save(di / "gt.png")
                m = track_a_metrics(gt, pred)
            else:
                m = track_b_metrics(inp, pred)
            row = {"track": track, "image": name, "variant": v, "seconds": dt, "metrics": m}
            per_image.append(row)
            print(f"[{track}] {name} / {v}: {m} ({dt}s)", flush=True)

    # Track A
    if args.track_a_manifest:
        manifest = json.loads(Path(args.track_a_manifest).read_text(encoding="utf-8"))
        if args.limit:
            manifest = manifest[:args.limit]
        for item in manifest:
            process("A", item["name"], load_rgb(item["gt"]))

    # Track B
    if args.track_b_dir:
        files = sorted(p for p in Path(args.track_b_dir).iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if args.track_b_list:
            allow = set(json.loads(Path(args.track_b_list).read_text(encoding="utf-8")))
            files = [p for p in files if p.stem in allow]
        if args.limit:
            files = files[:args.limit]
        for p in files:
            process("B", p.stem, load_rgb(p))

    merged = dict(prev)
    for r in per_image:
        merged[(r["track"], r["image"], r["variant"])] = r
    all_rows = list(merged.values())
    (outdir / "per_image.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summaries(outdir, all_rows, variants)

    print(f"\n[done] -> {outdir}", flush=True)


TRACK_METRICS = {
    "A": ["psnr", "ssim", "lpips", "dE00", "bw_dE00", "misplace_idx", "colorfulness"],
    "B": ["skin_blue_pct", "whole_blue_pct", "edge_align", "bleed_ratio",
          "colorfulness", "L_shift"],
}


def write_summaries(outdir: Path, all_rows: list[dict], variants: list[str]):
    for track, metrics in TRACK_METRICS.items():
        rows = [r for r in all_rows if r["track"] == track]
        if not rows:
            continue
        present = [v for v in variants if any(r["variant"] == v for r in rows)]
        summary = {v: {m: round(float(np.median(
            [r["metrics"][m] for r in rows if r["variant"] == v])), 3) for m in metrics}
            for v in present}
        base = "sdxl_s080_cfg7" if "sdxl_s080_cfg7" in present else present[0]
        stats = significance_table(rows, metrics, base, present)
        (outdir / f"summary_{track}.json").write_text(json.dumps(
            {"metrics": metrics, "baseline": base, "n_images": len({r["image"] for r in rows}),
             "medians": summary, "significance": stats}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        lines = [f"# Track {track} 中位数（baseline={base}）", "",
                 "| variant | " + " | ".join(metrics) + " |",
                 "|" + "---|" * (len(metrics) + 1)]
        for v in present:
            lines.append("| " + v + " | " + " | ".join(str(summary[v][m]) for m in metrics) + " |")
        (outdir / f"summary_{track}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)


def recompute_from_disk(outdir: Path, variants: list[str], work_res: int,
                        track_b_dir: str | None):
    """不跑模型，从已保存的 PNG 重算指标（修 bug/改口径后回填）。"""
    rows: list[dict] = []
    a_root = outdir / "A"
    if a_root.exists():
        for di in sorted(d for d in a_root.iterdir() if d.is_dir()):
            gt_p, in_p = di / "gt.png", di / "input_gray.png"
            if not gt_p.exists():
                continue
            gt = load_rgb(gt_p)
            for v in variants:
                p = di / f"{v}.png"
                if p.exists():
                    rows.append({"track": "A", "image": di.name, "variant": v,
                                 "seconds": None, "metrics": track_a_metrics(gt, load_rgb(p))})
    b_root = outdir / "B"
    if b_root.exists() and track_b_dir:
        srcmap = {p.stem: p for p in Path(track_b_dir).iterdir()
                  if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
        for di in sorted(d for d in b_root.iterdir() if d.is_dir()):
            p = srcmap.get(di.name)
            if not p:
                continue
            src = retouch.fit_for_work(load_rgb(p), work_res)
            for v in variants:
                pp = di / f"{v}.png"
                if pp.exists():
                    rows.append({"track": "B", "image": di.name, "variant": v,
                                 "seconds": None, "metrics": track_b_metrics(src, load_rgb(pp))})
    return rows


if __name__ == "__main__":
    main()
