#!/usr/bin/env python3
"""DDColor 最小复现（#226）：加载 DDColor-L 权重，测 256/512 推理耗时与显存。

配合 docs/ddcolor-原理调研.md 的「最小复现」节。结果用于核对论文参数量（227.9M）
与本机 P40 实测速度/显存，不涉及训练。

用法：
  CUDA_VISIBLE_DEVICES=0 python3 inference/photo/tools/repro_ddcolor_min.py \
      --repo /mnt/data/ai_workspace/DDColor \
      --weights /mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt \
      --out /tmp/ddcolor_repro/out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("DDCOLOR_REPO", "/mnt/data/ai_workspace/DDColor"))
    ap.add_argument("--weights", default=os.environ.get(
        "DDCOLOR_CKPT", "/mnt/data/ai_workspace/models/ddcolor/ddcolor_modelscope.pt"))
    ap.add_argument("--out", default="/tmp/ddcolor_repro/out")
    ap.add_argument("--test-dir", default="assets/test_images")
    args = ap.parse_args()

    sys.path.insert(0, args.repo)
    from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

    test_dir = args.test_dir if os.path.isabs(args.test_dir) else os.path.join(args.repo, args.test_dir)
    imgs = [os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))][:2]
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    results = []
    for input_size in (256, 512):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = build_ddcolor_model(DDColor, model_path=args.weights, input_size=input_size,
                                    model_size="large", device=device)
        load_s = time.time() - t0
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        pipe = ColorizationPipeline(model, input_size=input_size, device=device)
        _ = pipe.process(cv2.imread(imgs[0]))  # warmup
        rows = []
        for p in imgs:
            img = cv2.imread(p)
            t1 = time.time()
            out = pipe.process(img)
            dt = time.time() - t1
            name = os.path.basename(p)[:30].replace(" ", "_")
            cv2.imwrite(os.path.join(args.out, f"{name}_s{input_size}.png"), out)
            rows.append({"img": os.path.basename(p)[:40], "hxw": list(img.shape[:2]), "infer_s": round(dt, 3)})
            print(f"[s={input_size}] {os.path.basename(p)[:40]} {img.shape[:2]} {dt:.3f}s")
        r = {"input_size": input_size, "params_M": round(params_m, 1), "load_s": round(load_s, 2),
             "peak_mem_MiB": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1), "rows": rows}
        results.append(r)
        json.dump(r, open(f"/tmp/ddcolor_repro/res_{input_size}.json", "w"), ensure_ascii=False, indent=2)
    json.dump(results, open("/tmp/ddcolor_repro/results.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
