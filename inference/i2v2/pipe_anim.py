#!/usr/bin/env python3
"""管线 C：白底角色动画（16 帧 I2V + 逐帧 RMBG 抠图 → 透明视频序列）。

用法:
    python3 pipe_anim.py --sb storyboard.json --character character.json --prefix lumo
    python3 pipe_anim.py --sb ... --scene 3

输出:
    outputs_video/{prefix}/assets/anim_white/scene00X.mp4      （白底动画）
    outputs_video/{prefix}/assets/anim/scene00X_%03d.png       （透明帧序列）
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "i2v"))

from generate import http_json, WORKFLOW, INPUT_DIR  # noqa: E402

COMFY_URL = "http://127.0.0.1:10337"
OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"
FPS = 8


def anim_shot(frame_path: str, face_anchor: str, seed: int, frames: int = 16,
              steps: int = 20, denoise: float = 0.8) -> str:
    """白底角色 I2V → webp 路径"""
    import shutil
    from PIL import Image as PILImage
    # 首帧（白底角色图）→ input/i2v_ref.png
    img = PILImage.open(frame_path).convert("RGB").resize((512, 512), PILImage.LANCZOS)
    img.save(os.path.join(INPUT_DIR, "i2v_ref.png"))
    # 脸部锚定 → input/i2v_char.png
    ref = PILImage.open(face_anchor).convert("RGB").resize((512, 512), PILImage.LANCZOS)
    ref.save(os.path.join(INPUT_DIR, "i2v_char.png"))

    with open(WORKFLOW) as f:
        wf = json.load(f)
    wf["2"]["inputs"]["image"] = "i2v_ref.png"
    wf["14"]["inputs"]["image"] = "i2v_char.png"
    wf["4"]["inputs"]["amount"] = frames
    wf["11"]["inputs"].update(steps=steps, seed=seed, denoise=denoise)
    wf["13"]["inputs"].update(filename_prefix=f"anim_s{seed}", fps=FPS)
    pid = http_json(f"{COMFY_URL}/prompt", {"prompt": wf}, timeout=120)["prompt_id"]
    import time
    t0 = time.time()
    while True:
        time.sleep(5)
        h = http_json(f"{COMFY_URL}/history/{pid}", timeout=30)
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("completed"):
                for o in h[pid]["outputs"].values():
                    for img_ in o.get("images", []):
                        print(f"  [anim] I2V 完成 {time.time()-t0:.0f}s")
                        return f"/mnt/data/ai_workspace/ComfyUI/output/{img_['filename']}"
            if st.get("status_str") == "error":
                raise RuntimeError(f"动画失败: {st.get('messages',[st])[-1]}")
        if time.time() - t0 > 900:
            raise RuntimeError("动画超时")


def frames_to_alpha(webp: str, out_prefix: str, rmbg_net=None):
    """webp → 逐帧 PNG → RMBG 抠图透明序列"""
    from PIL import Image
    import numpy as np
    im = Image.open(webp)
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    for i in range(getattr(im, "n_frames", 1)):
        im.seek(i)
        rgb = im.convert("RGB")
        out = f"{out_prefix}_{i:03d}.png"
        if rmbg_net is not None:
            from char_sheet import seg_alpha
            mask = seg_alpha(rgb, rmbg_net)
            rgba = np.dstack((np.array(rgb), mask))
            Image.fromarray(rgba).save(out)
        else:
            rgb.save(out)
    print(f"  [anim] 透明帧序列: {os.path.dirname(out_prefix)}/ ({getattr(im, 'n_frames', 1)} 帧)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb", required=True)
    ap.add_argument("--character", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--seed", type=int, default=300)
    ap.add_argument("--skip-alpha", action="store_true")
    args = ap.parse_args()

    sb = json.load(open(args.sb))
    ch = json.load(open(args.character))
    char_dir = os.path.dirname(os.path.abspath(args.character))
    face_anchor = os.path.join(char_dir, "face_anchor.png")
    if not os.path.exists(face_anchor):
        face_anchor = os.path.join(char_dir, ch["files"]["front"])

    white_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "char_white")
    anim_white_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "anim_white")
    anim_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "anim")
    os.makedirs(anim_white_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    rmbg = None
    if not args.skip_alpha:
        sys.path.insert(0, os.path.join(BASE, "..", "i2v"))
        from char_sheet import BriaRMBG, RMBG_WEIGHTS
        print("[anim] 加载 RMBG-1.4...")
        rmbg = BriaRMBG.from_pretrained(RMBG_WEIGHTS).to("cuda").eval()

    for clip in sb["clips"]:
        s = clip["scene"]
        if args.scene and s != args.scene:
            continue
        frame = os.path.join(white_dir, f"scene{s:03d}.png")
        assert os.path.exists(frame), f"缺少白底角色图: {frame}（先跑 pipe_char）"
        anim_mp4 = os.path.join(anim_white_dir, f"scene{s:03d}.mp4")
        seq_ok = len([f for f in os.listdir(anim_dir) if f.startswith(f"scene{s:03d}_")]) >= 16
        if os.path.exists(anim_mp4) and (args.skip_alpha or seq_ok):
            print(f"  [anim] 已存在: scene{s:03d}")
            continue
        print(f"  [anim] 镜头{s}: I2V 出片中...")
        webp = anim_shot(frame, face_anchor, args.seed + s)
        # webp → mp4（白底动画，无音频）
        frames_to_alpha(webp, os.path.join(anim_dir, f"scene{s:03d}"), rmbg_net=rmbg)
        # 也存白底 mp4 供预览/对比
        import subprocess as sp
        sp.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i",
                os.path.join(anim_dir, f"scene{s:03d}_%03d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", anim_mp4],
               check=True, capture_output=True)
        os.remove(webp)
        print(f"  [anim] {anim_mp4}")


if __name__ == "__main__":
    main()
