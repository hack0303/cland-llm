#!/usr/bin/env python3
"""管线 B：背景生成 — 纯场景出图（无角色），质量独立拉满。

用法:
    python3 pipe_bg.py --sb outputs_video/lumo/storyboard.json --prefix lumo
    python3 pipe_bg.py --sb ... --scene 3            # 只出镜头 3
    python3 pipe_bg.py --sb ... --model sdxl          # 引擎切换（默认 Counterfeit）

输出: outputs_video/{prefix}/assets/bg/scene00X.png
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "i2v"))
from run_story import http_json, t2i_workflow  # noqa: E402

COMFY_URL = "http://127.0.0.1:10337"
OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"
STYLE_TAIL = "highly detailed, sharp focus, intricate, cinematic lighting"


def gen_bg(prompt: str, seed: int) -> str:
    """ComfyUI Counterfeit 出图（无锚定，纯场景）→ 返回输出文件路径"""
    wf = t2i_workflow(f"{prompt}, {STYLE_TAIL}", seed, char_image=None)
    wf["7"]["inputs"]["filename_prefix"] = "bg"
    pid = http_json(f"{COMFY_URL}/prompt", {"prompt": wf}, timeout=120)["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(4)
        h = http_json(f"{COMFY_URL}/history/{pid}", timeout=30)
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed"):
                for o in h[pid]["outputs"].values():
                    for img in o.get("images", []):
                        return f"/mnt/data/ai_workspace/ComfyUI/output/{img['filename']}"
            if st.get("status_str") == "error":
                raise RuntimeError(f"背景出图失败: {st.get('messages',[st])[-1]}")
        if time.time() - t0 > 180:
            raise RuntimeError("背景出图超时")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb", required=True, help="storyboard.json 路径")
    ap.add_argument("--prefix", required=True, help="输出目录名")
    ap.add_argument("--scene", type=int, default=0, help="只出指定镜头（0=全部）")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()

    sb = json.load(open(args.sb))
    out_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "bg")
    os.makedirs(out_dir, exist_ok=True)

    for clip in sb["clips"]:
        s = clip["scene"]
        if args.scene and s != args.scene:
            continue
        out = os.path.join(out_dir, f"scene{s:03d}.png")
        if os.path.exists(out):
            print(f"  [bg] 已存在: {out}")
            continue
        bg_prompt = clip.get("bg_prompt") or clip["image_prompt"]
        print(f"  [bg] 镜头{s}: 出图中...")
        src = gen_bg(bg_prompt, args.seed + s)
        shutil.move(src, out)
        print(f"  [bg] {out}")


if __name__ == "__main__":
    main()
