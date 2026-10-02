#!/usr/bin/env python3
"""#225 D 路：DDColor 专用上色模型接入（对拍用）。

用 DDColor（ImageNet 预训练，modelscope 权重）对同一 work 图出色，产出：
  - D_native    ：DDColor 原生输出（自带亮度）
  - D_transfer  ：DDColor 出色只取色度、盖到 work 亮度（与产线 chroma_transfer 口径一致）

需外部 DDColor 仓库与权重：
  --ddcolor-repo /tmp/DDColor --model-path /tmp/ddcolor_w/pytorch_model.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS))
import cmp_colorize_offset as C  # noqa: E402

sys.path.insert(0, str(_THIS.parent))
import retouch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ddcolor-repo", default="/tmp/DDColor")
    ap.add_argument("--model-path", default="/tmp/ddcolor_w/pytorch_model.pt")
    ap.add_argument("--zoom", default="0.45,0.45,0.85,0.95")
    args = ap.parse_args()

    sys.path.insert(0, args.ddcolor_repo)
    import torch
    from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    box = tuple(float(v) for v in args.zoom.split(","))

    orig = retouch.load_rgb(args.input)
    work = C.prepare(orig)
    print(f"[DDColor] work={work.size}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ddcolor_model(DDColor, model_path=args.model_path,
                                input_size=512, model_size="large", device=device)
    pipe = ColorizationPipeline(model, input_size=512, device=device)

    bgr = cv2.cvtColor(np.asarray(work.convert("RGB")), cv2.COLOR_RGB2BGR)
    out_bgr = pipe.process(bgr)
    native = Image.fromarray(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
    transfer = retouch.chroma_transfer(native, work, 1.0)

    rows = []
    for name, after, note in [("D1_ddcolor_native", native, "DDColor 原生（自带亮度）"),
                              ("D2_ddcolor_transfer", transfer, "DDColor 色度→work 亮度（对拍口径）")]:
        d = outdir / name
        d.mkdir(exist_ok=True)
        after.save(d / "after.png")
        met = C.metrics(work, after)
        met.update({"variant": name, "route": "D-专用模型", "note": note})
        # 蓝块指标
        lab = cv2.cvtColor(np.asarray(after.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
        a = lab[..., 1] - 128
        b = lab[..., 2] - 128
        sat = np.hypot(a, b)
        hue = np.degrees(np.arctan2(b, a))
        met["blue_artifact_pct"] = round(float(((sat > 18) & (hue > -140) & (hue < -40)).mean() * 100), 3)
        rows.append(met)
        C.compose(work, after, f"{name} | D-专用模型 | align={met['edge_align']} "
                  f"blue%={met['blue_artifact_pct']} Lshift={met['L_shift']} | {note}",
                  d / "compare.png", box)
        (d / "meta.json").write_text(json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[done] {name} {met}", flush=True)

    (outdir / "summary.json").write_text(
        json.dumps({"input": args.input, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
