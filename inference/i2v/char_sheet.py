#!/usr/bin/env python3
"""角色设定工作流 — prompt-hub 渲染提示词（唯一真源）+ 出图 + RMBG 抠图 + 资产打包。

三阶段（对齐动画公司标准流程）:
    Step 1 角色母版: 三视图（front/side/back 单视图 + 拼版，档案）
    Step 2 动作资产: 正面定妆 + 动作/表情（prompt-hub 渲染 → SDXL → RMBG 透明 PNG）
    Step 3 视频接线: front.png 即 run_story 的 IPAdapter 锚定图（--character）

提示词来源: prompt-hub CLI（/home/alice/work/agentic/alice-prompt-hub）
    模板: {desc}, {style}, {lighting}, 中间段, highly detailed, sharp focus,
          intricate, plain background, white, single character

用法:
    python3 char_sheet.py --name lumo \
        --desc "a tiny round glowing firefly fairy with a warm yellow cloak..." \
        --style "cute healing fantasy" \
        --lighting "soft volumetric glow, warm rim light" \
        --actions WALKING,HOLDING,WAVING \
        --expressions HAPPY,SAD,SURPRISED

输出: /mnt/data/ai_workspace/outputs_character/{name}/
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
TRIOSG_SCRIPTS = os.path.join(BASE, "..", "triposg", "scripts")
sys.path.insert(0, os.path.abspath(TRIOSG_SCRIPTS))

import numpy as np
import cv2
import torch
from PIL import Image
from skimage.morphology import remove_small_objects
from skimage.measure import label
from briarmbg import BriaRMBG

SDXL_URL = "http://127.0.0.1:10331"
RMBG_WEIGHTS = "/mnt/data/ai_workspace/cland-llm/inference/triposg/pretrained_weights/RMBG-1.4"
OUT_ROOT = "/mnt/data/ai_workspace/outputs_character"
PROMPT_HUB_DIR = "/home/alice/work/agentic/alice-prompt-hub"
DEFAULT_NEGATIVE = "blurry, low quality, distorted, watermark, text"

# 中文兼容映射 → prompt-hub 枚举（推荐直接用英文枚举）
CN_ACTIONS = {"挥手": "WAVING", "奔跑": "RUNNING", "跳跃": "FLYING", "飞行": "FLYING",
              "坐下": "SITTING", "跪地": "KNEELING", "舞蹈": "DANCING", "手持": "HOLDING", "行走": "WALKING"}
CN_EXPRESSIONS = {"微笑": "HAPPY", "悲伤": "SAD", "惊讶": "SURPRISED", "生气": "ANGRY",
                  "害羞": "SHY", "自信": "DETERMINED", "平静": "CALM", "兴奋": "EXCITED"}
CN_VIEWS = {"正面": "FRONT", "侧面": "SIDE", "背面": "BACK"}


def prompt_hub_sheet(name: str, desc: str, style: str, lighting: str,
                     views: list[str], actions: list[str], expressions: list[str]) -> dict:
    """调用 prompt-hub 渲染角色资产提示词（唯一真源）。"""
    def norm(items, mapping):
        return ",".join(mapping.get(i, i.upper()) for i in items)
    cmd = ["uv", "run", "prompt-hub", "character",
           "--name", name, "--desc", desc, "--style", style, "--lighting", lighting,
           "--views", norm(views, CN_VIEWS),
           "--actions", norm(actions, CN_ACTIONS),
           "--expressions", norm(expressions, CN_EXPRESSIONS)]
    r = subprocess.run(cmd, cwd=PROMPT_HUB_DIR, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"prompt-hub 渲染失败: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def sdxl(prompt: str, seed: int, out: str, negative: str = DEFAULT_NEGATIVE):
    if os.path.exists(out):
        print(f"  [sdxl] 已存在: {out}")
        return
    req = urllib.request.Request(SDXL_URL + "/generate",
                                 data=json.dumps({"prompt": prompt, "steps": 30, "seed": seed,
                                                  "negative_prompt": negative}).encode(),
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
    resize = torch.nn.functional.interpolate(rgb_gpu.unsqueeze(0), size=(1024, 1024), mode="bilinear").squeeze(0)
    max_v = resize.flatten().max()
    norm = resize / max_v - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
    with torch.no_grad():
        alpha = rmbg_net(norm.unsqueeze(0))[0][0]  # [1,1,1024,1024]
    alpha = torch.nn.functional.interpolate(alpha, size=(img.shape[0], img.shape[1]), mode="bilinear").squeeze()
    ma, mi = alpha.max(), alpha.min()
    alpha = (alpha - mi) / (ma - mi)
    alpha_np = (alpha * 255).to(torch.uint8).cpu().numpy()
    _, alpha_np = cv2.threshold(alpha_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = remove_small_objects(label(alpha_np) > 0, min_size=200).astype(np.uint8) * 255
    return cleaned


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


def _is_true_back(path: str, threshold: float = 0.03) -> bool:
    """背面方向验证：面部区域肤色占比 <3% 视为真背面（无正脸）"""
    try:
        import numpy as _np
        im = _np.array(Image.open(path).convert("RGB")).astype(float)
        h, w, _ = im.shape
        face = im[int(h * 0.2):int(h * 0.5), int(w * 0.3):int(w * 0.7)]
        skin = (face[:, :, 0] > face[:, :, 1] + 15) & (face[:, :, 1] > face[:, :, 2])
        return skin.mean() < threshold
    except Exception:
        return True  # 无法验证时保留


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="角色名（目录名，如 lumo）")
    ap.add_argument("--desc", required=True, help="角色描述（英文，跨镜头固定复用；含脚部/鞋细节）")
    ap.add_argument("--style", default="cute healing fantasy", help="风格短语（英文）")
    ap.add_argument("--lighting", default="soft volumetric glow, warm rim light", help="光照段（英文）")
    ap.add_argument("--views", nargs="*", default=["FRONT"],
                    help="视角: FRONT（默认，生产必需） SIDE BACK（可选档案；BACK 方向不可靠，自动验证失败即跳过）")
    ap.add_argument("--actions", nargs="*", default=[], help="动作: WALKING RUNNING FLYING KNEELING DANCING HOLDING WAVING SITTING（或中文）")
    ap.add_argument("--expressions", nargs="*", default=[], help="表情: HAPPY SAD SURPRISED ANGRY SHY DETERMINED CALM EXCITED（或中文）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-3view", action="store_true", help="跳过三视图")
    ap.add_argument("--no-alpha", action="store_true", help="跳过抠图")
    args = ap.parse_args()

    print("[char] 调用 prompt-hub 渲染提示词...")
    ph = prompt_hub_sheet(args.name, args.desc, args.style, args.lighting,
                          args.views, args.actions, args.expressions)
    print(f"[char] 渲染完成: {len(ph['trisheet'])} 视图 + 定妆 + {len(ph['actions'])} 动作 + {len(ph['expressions'])} 表情")

    out_dir = os.path.join(OUT_ROOT, args.name)
    os.makedirs(out_dir, exist_ok=True)
    assets = {"name": args.name, "desc": args.desc, "style": args.style,
              "lighting": args.lighting, "files": {}}

    # Step 1: 三视图（单视图 + 拼版；BACK 方向不可靠 → 肤色检测验证，失败跳过并标注）
    view_files = []
    if not args.no_3view:
        for i, (view, prompt) in enumerate(ph["trisheet"].items()):
            vp = os.path.join(out_dir, "views", f"{view}.png")
            sdxl(prompt, args.seed + 10 + i, vp)
            if not os.path.exists(vp):
                continue
            # BACK 方向验证：面部区肤色占比 >3% = 露脸（SDXL 常把背面画成正面）→ 跳过
            if view == "back" and not _is_true_back(vp):
                print(f"  [view] {view} 方向验证失败（露脸），跳过（能力边界，见 CHARACTER.md）")
                os.remove(vp)
                continue
            assets["files"][f"view_{view}"] = os.path.relpath(vp, out_dir)
            view_files.append(vp)
        if view_files:
            sheet = os.path.join(out_dir, "character_sheet_3view.png")
            if not os.path.exists(sheet):
                ims = [Image.open(p).convert("RGB") for p in view_files]
                cell = 1024 // len(ims)
                canvas = Image.new("RGB", (1024, 1024), "white")
                for j, im in enumerate(ims):
                    canvas.paste(im.resize((cell, 1024), Image.LANCZOS), (j * cell, 0))
                canvas.save(sheet)
            print(f"  [sheet] {sheet}")
            assets["files"]["sheet_3view"] = "character_sheet_3view.png"

    # Step 2a: 正面定妆（I2V 锚定图）
    front = os.path.join(out_dir, "front.png")
    sdxl(ph["fullbody"], args.seed + 1, front)
    assets["files"]["front"] = "front.png"

    # Step 2b: 动作资产
    for i, (action, prompt) in enumerate(ph["actions"].items()):
        p = os.path.join(out_dir, "poses", f"{action}.png")
        sdxl(prompt, args.seed + 2 + i, p)
        assets["files"][f"action_{action}"] = os.path.relpath(p, out_dir)

    # Step 2c: 表情资产（特写）
    for i, (expr, prompt) in enumerate(ph["expressions"].items()):
        p = os.path.join(out_dir, "expressions", f"{expr}.png")
        sdxl(prompt, args.seed + 20 + i, p)
        assets["files"][f"expr_{expr}"] = os.path.relpath(p, out_dir)

    # Step 3: RMBG 抠图
    if not args.no_alpha:
        print("[char] 加载 RMBG-1.4 抠图模型...")
        rmbg_net = BriaRMBG.from_pretrained(RMBG_WEIGHTS).to("cuda").eval()
        to_alpha_png(front, os.path.join(out_dir, "front_alpha.png"), rmbg_net)
        assets["files"]["front_alpha"] = "front_alpha.png"  # ← 视频锚定（注：I2V 用白底 front.png）
        for action in ph["actions"]:
            p = os.path.join(out_dir, "poses", f"{action}.png")
            if os.path.exists(p):
                to_alpha_png(p, os.path.join(out_dir, "poses", f"{action}_alpha.png"), rmbg_net)
                assets["files"][f"action_{action}_alpha"] = os.path.relpath(os.path.join(out_dir, "poses", f"{action}_alpha.png"), out_dir)
        for expr in ph["expressions"]:
            p = os.path.join(out_dir, "expressions", f"{expr}.png")
            if os.path.exists(p):
                to_alpha_png(p, os.path.join(out_dir, "expressions", f"{expr}_alpha.png"), rmbg_net)
                assets["files"][f"expr_{expr}_alpha"] = os.path.relpath(os.path.join(out_dir, "expressions", f"{expr}_alpha.png"), out_dir)

    meta = os.path.join(out_dir, "character.json")
    with open(meta, "w") as f:
        json.dump(assets, f, ensure_ascii=False, indent=2)

    print(f"[char] 完成: {out_dir}")
    print(f"[char] 视频锚定图（run_story --character {meta}）: front.png（白底）")


if __name__ == "__main__":
    main()
