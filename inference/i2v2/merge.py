#!/usr/bin/env python3
"""管线 E：合成 — 透明角色动画 overlay 到背景 + 时长补帧 + 音频混流。

用法:
    python3 merge.py --sb storyboard.json --prefix lumo [--pos center] [--scale 1.0]
    python3 merge.py --sb ... --scene 3

流程（逐镜头）:
    1. 透明角色帧序列 overlay 到背景图 → 合成镜头片段（tpad 补到 duration）
    2. 全部镜头 → compose.py 拼接 + 配音混流 → {prefix}_final.mp4

输出:
    outputs_video/{prefix}/assets/merged/scene00X.mp4
    outputs_video/{prefix}/{prefix}_final.mp4
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"
FPS = 8


def overlay_shot(bg: str, anim_seq: str, out: str, duration: float, pos: str = "center", scale: float = 1.0):
    """透明角色序列 overlay 背景 → 补帧到 duration → mp4"""
    # 位置计算（背景 512×512，角色图 512×512 透明，角色居中）
    if pos == "center":
        x, y = 128, 128  # 角色图缩放到 256 时居中偏移
    else:
        x, y = 0, 0
    # scale：角色图缩放（512 → 512*scale），overlay 位置相应调整
    sw = int(512 * scale)
    ox = int((512 - sw) / 2) if pos == "center" else x
    oy = int((512 - sw) / 2) if pos == "center" else y
    cmd = ["ffmpeg", "-y", "-i", bg,
           "-framerate", str(FPS), "-i", anim_seq,
           "-filter_complex",
           f"[1:v]scale={sw}:{sw},format=rgba[ch];"
           f"[0:v][ch]overlay={ox}:{oy},format=yuv420p",
           "-frames:v", str(max(16, int(round(duration * FPS)))),
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS), out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"overlay 失败: {r.stderr[-500:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--pos", default="center", help="角色位置: center/bottom-left/...")
    ap.add_argument("--scale", type=float, default=1.0, help="角色缩放（0.5-1.5）")
    ap.add_argument("--no-audio", action="store_true", help="跳过音频（只合成画面）")
    args = ap.parse_args()

    sb = json.load(open(args.sb))
    assets = os.path.join(OUT_ROOT, args.prefix, "assets")
    bg_dir = os.path.join(assets, "bg")
    anim_dir = os.path.join(assets, "anim")
    merged_dir = os.path.join(assets, "merged")
    os.makedirs(merged_dir, exist_ok=True)

    merged_clips = []
    for clip in sb["clips"]:
        s = clip["scene"]
        if args.scene and s != args.scene:
            continue
        bg = os.path.join(bg_dir, f"scene{s:03d}.png")
        seq = os.path.join(anim_dir, f"scene{s:03d}_%03d.png")
        out = os.path.join(merged_dir, f"scene{s:03d}.mp4")
        if not os.path.exists(out):
            print(f"  [merge] 镜头{s}: overlay 合成...")
            overlay_shot(bg, seq, out, clip["duration"], args.pos, args.scale)
        else:
            print(f"  [merge] 已存在: scene{s:03d}")
        merged_clips.append(out)

    if args.no_audio or not any(clip.get("voice") for clip in sb["clips"]):
        # 无音频：直接拼接
        lst = os.path.join(merged_dir, "list.txt")
        with open(lst, "w") as f:
            for c in merged_clips:
                f.write(f"file '{c}'\n")
        final = os.path.join(OUT_ROOT, args.prefix, f"{args.prefix}_final.mp4")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", final], check=True, capture_output=True)
        print(f"[merge] 输出（无音频）: {final}")
        return

    # 有音频：调 compose.py（复用音频链/时间轴/tpad）
    voice_args, durations, t_global = [], [], 0.0
    for clip in sb["clips"]:
        s = clip["scene"]
        v = clip.get("voice")
        v_len = 0.0
        if v:
            wav = os.path.join(OUT_ROOT, args.prefix, "assets", "audio", f"scene{s:03d}_voice.wav")
            if os.path.exists(wav):
                import re
                r = subprocess.run(["ffmpeg", "-i", wav, "-f", "null", "-"], capture_output=True, text=True)
                m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
                if m:
                    v_len = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
                voice_args += ["--voice", wav, "--voice-at", str(round(t_global + clip["voice_at"], 2))]
        durations.append(round(max(clip["duration"], clip["voice_at"] + v_len + 0.3), 2))
        t_global += durations[-1]

    subprocess.run([sys.executable, os.path.join(BASE, "..", "i2v", "compose.py"),
                    "--clips"] + merged_clips + voice_args +
                   ["--durations"] + [str(d) for d in durations] +
                   ["--prefix", f"{args.prefix}_final",
                    "--outdir", os.path.join(OUT_ROOT, args.prefix)],
                   check=True)
    print(f"[merge] 输出: {os.path.join(OUT_ROOT, args.prefix, args.prefix + '_final.mp4')}")


if __name__ == "__main__":
    main()
