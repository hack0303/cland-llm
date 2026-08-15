#!/usr/bin/env python3
"""图生视频（AnimateDiff）生成脚本 — 通过 ComfyUI API 提交工作流并取回结果。

用法:
    python3 generate.py --image ref.png [--frames 16] [--steps 20] [--denoise 0.8]
                        [--seed 42] [--fps 8] [--prefix i2v_out] [--comfy http://127.0.0.1:10337]

流程: 参考图 → ComfyUI input/ → 提交 workflow → 轮询 history → 输出 webp → ffmpeg 转 mp4
"""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

WORKFLOW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_i2v.json")
INPUT_DIR = "/mnt/data/ai_workspace/ComfyUI/input"
OUTPUT_DIR = "/mnt/data/ai_workspace/outputs_video"


def http_json(url, data=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="参考图路径")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--denoise", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--prefix", default="i2v_out")
    ap.add_argument("--comfy", default="http://127.0.0.1:10337")
    ap.add_argument("--keep-webp", action="store_true", help="保留 webp 中间产物")
    args = ap.parse_args()

    # 1. 参考图放入 ComfyUI input/
    img_name = os.path.basename(args.image)
    shutil.copy(args.image, os.path.join(INPUT_DIR, img_name))
    if img_name != "i2v_ref.png":
        shutil.copy(args.image, os.path.join(INPUT_DIR, "i2v_ref.png"))

    # 2. 载入并改写 workflow
    with open(WORKFLOW) as f:
        wf = json.load(f)
    wf["2"]["inputs"]["image"] = "i2v_ref.png"
    wf["4"]["inputs"]["amount"] = args.frames
    wf["11"]["inputs"].update(steps=args.steps, seed=args.seed, cfg=args.cfg, denoise=args.denoise)
    wf["13"]["inputs"].update(filename_prefix=args.prefix, fps=args.fps)

    # 3. 提交
    t0 = time.time()
    pid = http_json(f"{args.comfy}/prompt", {"prompt": wf}, timeout=120)["prompt_id"]
    print(f"[i2v] 已提交 prompt_id={pid} frames={args.frames} steps={args.steps} denoise={args.denoise}")

    # 4. 轮询结果
    while True:
        time.sleep(5)
        try:
            h = http_json(f"{args.comfy}/history/{pid}", timeout=30)
        except Exception:
            continue
        if pid in h:
            status = h[pid].get("status", {})
            if status.get("completed"):
                outs = h[pid]["outputs"]
                # 找 SaveAnimatedWEBP 节点（13）输出
                for nid, o in outs.items():
                    for img in o.get("images", []):
                        if img.get("type") == "output":
                            print(f"[i2v] 完成，耗时 {time.time()-t0:.0f}s")
                            return finalize(img["filename"], img["subfolder"], args)
            if status.get("status_str") == "error":
                msgs = status.get("messages", [])
                print(f"[i2v] 失败: {msgs[-1] if msgs else status}", file=sys.stderr)
                sys.exit(1)


def finalize(fname, subfolder, args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src = os.path.join(OUTPUT_DIR, subfolder, fname) if subfolder else os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(src):  # 兼容 output 目录配置
        src = os.path.join("/mnt/data/ai_workspace/ComfyUI/output", subfolder, fname)
    mp4 = os.path.join(OUTPUT_DIR, f"{args.prefix}.mp4")
    # 动画 webp → PIL 提取帧 → ffmpeg 拼 mp4（ffmpeg 直接解动画 webp 会失败，实测 exit 69）
    import tempfile
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        im = Image.open(src)
        for i in range(getattr(im, "n_frames", 1)):
            im.seek(i)
            im.save(os.path.join(td, f"f{i:03d}.png"))
        subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i",
                        os.path.join(td, "f%03d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        mp4], check=True, capture_output=True)
    if not args.keep_webp:
        os.remove(src)
    print(f"[i2v] 输出: {mp4}")


if __name__ == "__main__":
    main()
