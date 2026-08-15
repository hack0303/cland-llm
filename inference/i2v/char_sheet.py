#!/usr/bin/env python3
"""角色设定工作流 — 三视图母版 + 动作资产（透明 PNG）+ 视频锚定接线。

三阶段（对齐动画公司标准流程）:
    Step 1 角色母版: 三视图（档案/IP身份证，白底保留原图）
    Step 2 动作资产: 正面全身 + 各姿态（SDXL 生成 → RMBG-1.4 抠图 → 透明 PNG）
    Step 3 视频接线: front_alpha.png 即 run_story 的 IPAdapter 锚定图（--character）

用法:
    python3 char_sheet.py --name lumo \
        --desc "a cute firefly fairy with a glowing chest core and purple cloak" \
        --style "cute cartoon, soft lighting" \
        --poses standing sitting running smiling

输出: /mnt/data/ai_workspace/outputs_character/{name}/
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
TRIOSG_SCRIPTS = os.path.join(BASE, "..", "triposg", "scripts")
sys.path.insert(0, os.path.abspath(TRIOSG_SCRIPTS))

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.morphology import remove_small_objects
from skimage.measure import label
from briarmbg import BriaRMBG

SDXL_URL = "http://127.0.0.1:10331"
RMBG_WEIGHTS = "/mnt/data/ai_workspace/cland-llm/inference/triposg/pretrained_weights/RMBG-1.4"
OUT_ROOT = "/mnt/data/ai_workspace/outputs_character"

# 动作表：中文名 → 英文提示词片段（动画动作）
POSES = {
    "待机": "idle standing pose, facing forward, relaxed",
    "站立": "standing pose, facing forward",
    "行走": "walking pose",
    "奔跑": "running pose, dynamic",
    "跳跃": "jumping pose, mid-air",
    "攻击": "attacking pose, weapon swing",
    "受击": "hit pose, knocked back",
    "施法": "casting pose, hands glowing with magic",
    "坐下": "sitting pose",
    "躺下": "lying down pose",
    "胜利": "victory pose, arms raised cheering",
    "失败": "defeated pose, sitting on ground tired",
    "挥手": "waving one hand, friendly",
    "思考": "thinking pose, hand on chin",
}

# 表情表：中文名 → 英文表情（情绪特写）
EXPRESSIONS = {
    "微笑": "smiling, happy",
    "惊讶": "surprised, eyes wide open",
    "生气": "angry, frowning",
    "悲伤": "sad, crying slightly",
    "害羞": "shy, blushing",
    "自信": "confident, determined",
}

# 三视图：视图名 → 英文提示词片段
VIEWS = {
    "front": "front view, facing forward",
    "side": "side view, profile facing left",
    "back": "back view, seen from behind",
}


def sdxl(prompt: str, seed: int, out: str):
    if os.path.exists(out):
        print(f"  [sdxl] 已存在: {out}")
        return
    req = urllib.request.Request(SDXL_URL + "/generate",
                                 data=json.dumps({"prompt": prompt, "steps": 30, "seed": seed}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        img = json.loads(r.read())["image"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.rename(img, out)
    print(f"  [sdxl] {out}")


def seg_alpha(img_pil: Image.Image, rmbg_net) -> np.ndarray:
    """RMBG-1.4 抠图 → alpha mask（复用 triposg load_image 逻辑）"""
    img = np.array(img_pil.convert("RGB"))
    rgb_gpu = torch.from_numpy(img).cuda().float().permute(2, 0, 1) / 255.
    resize = torchvision_resize(rgb_gpu, 1024)
    max_v = resize.flatten().max()
    norm = resize / max_v - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    with torch.no_grad():
        alpha = rmbg_net(norm.unsqueeze(0))[0][0]  # [1,1,1024,1024]
    alpha = torch.nn.functional.interpolate(alpha,  # 保持 [1,1,H,W]
                                            size=(img.shape[0], img.shape[1]),
                                            mode="bilinear").squeeze()
    ma, mi = alpha.max(), alpha.min()
    alpha = (alpha - mi) / (ma - mi)
    alpha_np = (alpha * 255).to(torch.uint8).cpu().numpy()
    _, alpha_np = cv2.threshold(alpha_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = remove_small_objects(label(alpha_np) > 0, min_size=200).astype(np.uint8) * 255
    return cleaned


def torchvision_resize(t, size):
    return torch.nn.functional.interpolate(t.unsqueeze(0), size=(size, size), mode="bilinear").squeeze(0)


def to_alpha_png(src: str, dst: str, rmbg_net):
    if os.path.exists(dst):
        print(f"  [alpha] 已存在: {dst}")
        return
    img = Image.open(src).convert("RGB")
    mask = seg_alpha(img, rmbg_net)
    rgba = np.dstack((np.array(img), mask))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.fromarray(rgba).save(dst)
    print(f"  [alpha] {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="角色名（目录名，如 lumo）")
    ap.add_argument("--desc", required=True, help="角色描述（英文，跨镜头固定复用的提示词短语）")
    ap.add_argument("--style", default="cute cartoon, soft lighting, vibrant colors", help="画风（英文）")
    ap.add_argument("--poses", nargs="*", default=[], help="动作列表（中文，见 POSES 表）")
    ap.add_argument("--expressions", nargs="*", default=[], help="表情列表（中文，见 EXPRESSIONS 表，特写）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-3view", action="store_true", help="跳过三视图（单视图三张 + 拼版）")
    ap.add_argument("--no-alpha", action="store_true", help="跳过抠图（只出白底原图）")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    pose_dir = os.path.join(out_dir, "poses")
    assets = {"name": args.name, "desc": args.desc, "style": args.style, "files": {}}

    # Step 1: 三视图母版（三张单视图 + 拼版；单张生成可控，避免 SDXL 一图三视图失控）
    if not args.no_3view:
        views_dir = os.path.join(out_dir, "views")
        os.makedirs(views_dir, exist_ok=True)
        view_files = []
        for i, (view, en) in enumerate(VIEWS.items()):
            vp = os.path.join(views_dir, f"{view}.png")
            sdxl(f"{args.desc}, {args.style}, full body, {en}, plain white background, "
                 f"single character, single view, high quality",
                 args.seed + 10 + i, vp)
            assets["files"][f"view_{view}"] = os.path.relpath(vp, out_dir)
            view_files.append(vp)
        # 拼版：三张等宽横排 → character_sheet_3view.png
        sheet = os.path.join(out_dir, "character_sheet_3view.png")
        if not os.path.exists(sheet):
            ims = [Image.open(p).convert("RGB") for p in view_files]
            w = 1024
            cell = w // 3
            canvas = Image.new("RGB", (w, 1024), "white")
            for j, im in enumerate(ims):
                im2 = im.resize((cell, 1024), Image.LANCZOS)
                canvas.paste(im2, (j * cell, 0))
            canvas.save(sheet)
        print(f"  [sheet] {sheet}")
        assets["files"]["sheet_3view"] = os.path.relpath(sheet, out_dir)

    # Step 2a: 正面全身定妆（生产锚定图）
    front = os.path.join(out_dir, "front.png")
    sdxl(f"{args.desc}, {args.style}, full body, front view, standing pose, plain white background, "
         f"single character, high quality",
         args.seed + 1, front)
    assets["files"]["front"] = os.path.relpath(front, out_dir)

    # Step 2b: 动作资产（全身）
    for i, pose in enumerate(args.poses):
        en = POSES.get(pose, pose)
        p = os.path.join(pose_dir, f"{pose}.png")
        sdxl(f"{args.desc}, {args.style}, full body, {en}, plain white background, single character, high quality",
             args.seed + 2 + i, p)
        assets["files"][f"pose_{pose}"] = os.path.relpath(p, out_dir)

    # Step 2c: 表情资产（脸部特写）
    expr_dir = os.path.join(out_dir, "expressions")
    for i, expr in enumerate(args.expressions):
        en = EXPRESSIONS.get(expr, expr)
        p = os.path.join(expr_dir, f"{expr}.png")
        os.makedirs(expr_dir, exist_ok=True)
        sdxl(f"{args.desc}, {args.style}, close-up portrait, {en} facial expression, "
             f"face filling the frame, plain white background, high quality",
             args.seed + 20 + i, p)
        assets["files"][f"expr_{expr}"] = os.path.relpath(p, out_dir)

    # Step 3: RMBG 抠图 → 透明 PNG（全部白底图）
    if not args.no_alpha:
        print("[char] 加载 RMBG-1.4 抠图模型...")
        rmbg_net = BriaRMBG.from_pretrained(RMBG_WEIGHTS).to("cuda").eval()
        to_alpha_png(front, os.path.join(out_dir, "front_alpha.png"), rmbg_net)
        assets["files"]["front_alpha"] = "front_alpha.png"  # ← 视频锚定图
        for pose in args.poses:
            p = os.path.join(pose_dir, f"{pose}.png")
            if os.path.exists(p):
                to_alpha_png(p, os.path.join(pose_dir, f"{pose}_alpha.png"), rmbg_net)
                assets["files"][f"pose_{pose}_alpha"] = os.path.relpath(os.path.join(pose_dir, f"{pose}_alpha.png"), out_dir)
        for expr in args.expressions:
            p = os.path.join(expr_dir, f"{expr}.png")
            if os.path.exists(p):
                to_alpha_png(p, os.path.join(expr_dir, f"{expr}_alpha.png"), rmbg_net)
                assets["files"][f"expr_{expr}_alpha"] = os.path.relpath(os.path.join(expr_dir, f"{expr}_alpha.png"), out_dir)

    meta = os.path.join(out_dir, "character.json")
    with open(meta, "w") as f:
        json.dump(assets, f, ensure_ascii=False, indent=2)

    print(f"[char] 完成: {out_dir}")
    print(f"[char] 视频锚定图（run_story --character {meta}）: front_alpha.png")


if __name__ == "__main__":
    main()
