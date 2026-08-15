#!/usr/bin/env python3
"""管线 A：角色白底出图（脸部锚定）→ RMBG 透明图。

用法:
    python3 pipe_char.py --sb storyboard.json --character character.json --prefix lumo
    python3 pipe_char.py --sb ... --scene 3

输出:
    outputs_video/{prefix}/assets/char_white/scene00X.png   （白底，锚定完美）
    outputs_video/{prefix}/assets/char_alpha/scene00X.png   （透明，合成用）
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
sys.path.insert(0, os.path.join(BASE, "..", "triposg", "scripts"))

import numpy as np  # noqa
import cv2  # noqa
import torch  # noqa
from PIL import Image  # noqa
from run_story import http_json, t2i_workflow, render_shot_prompt  # noqa
from char_sheet import BriaRMBG, RMBG_WEIGHTS, to_alpha_png  # noqa

COMFY_URL = "http://127.0.0.1:10337"
OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"
WHITE_TAIL = "standing on a plain white background, full body, single character"


def gen_char_white(prompt: str, seed: int, face_anchor: str) -> str:
    """白底角色出图（脸部锚定）→ 输出文件路径"""
    wf = t2i_workflow(f"{prompt}, {WHITE_TAIL}", seed, face_anchor, anchor=True)
    wf["7"]["inputs"]["filename_prefix"] = "charw"
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
                raise RuntimeError(f"角色出图失败: {st.get('messages',[st])[-1]}")
        if time.time() - t0 > 180:
            raise RuntimeError("角色出图超时")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb", required=True)
    ap.add_argument("--character", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--seed", type=int, default=200)
    ap.add_argument("--skip-alpha", action="store_true", help="跳过 RMBG 抠图")
    args = ap.parse_args()

    sb = json.load(open(args.sb))
    ch = json.load(open(args.character))
    char_dir = os.path.dirname(os.path.abspath(args.character))
    face_anchor = os.path.join(char_dir, "face_anchor.png")
    if not os.path.exists(face_anchor):
        face_anchor = os.path.join(char_dir, ch["files"]["front"])
    style = sb["style"]
    lighting = sb.get("lighting") or ch.get("lighting", "")

    white_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "char_white")
    alpha_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "char_alpha")
    os.makedirs(white_dir, exist_ok=True)
    os.makedirs(alpha_dir, exist_ok=True)

    rmbg = None
    if not args.skip_alpha:
        print("[char] 加载 RMBG-1.4...")
        rmbg = BriaRMBG.from_pretrained(RMBG_WEIGHTS).to("cuda").eval()

    for clip in sb["clips"]:
        s = clip["scene"]
        if args.scene and s != args.scene:
            continue
        white = os.path.join(white_dir, f"scene{s:03d}.png")
        alpha = os.path.join(alpha_dir, f"scene{s:03d}.png")
        if os.path.exists(white) and (args.skip_alpha or os.path.exists(alpha)):
            print(f"  [char] 已存在: scene{s:03d}")
            continue
        prompt = render_shot_prompt(clip, style, lighting)
        print(f"  [char] 镜头{s}: 白底出图...")
        src = gen_char_white(prompt, args.seed + s, face_anchor)
        shutil.move(src, white)
        print(f"  [char] {white}")
        if rmbg is not None:
            to_alpha_png(white, alpha, rmbg)
            print(f"  [char] {alpha}")


if __name__ == "__main__":
    main()
