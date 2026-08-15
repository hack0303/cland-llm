#!/usr/bin/env python3
"""V2 编排：多管线调度（幂等断点，每管线可独立/可选执行）。

用法:
    python3 pipeline.py --sb storyboard.json --character character.json --prefix lumo
    python3 pipeline.py --sb ... --only bg,char    # 只跑指定管线
    python3 pipeline.py --sb ... --scene 3          # 只处理镜头 3
"""
import argparse
import json
import os
import subprocess
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"
PIPES = ["bg", "char", "anim", "audio", "merge"]


def run(cmd):
    print(f"[pipe] {' '.join(cmd[:3])}...")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"[pipe] 失败: {' '.join(cmd[:4])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb", required=True)
    ap.add_argument("--character", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--only", default=",".join(PIPES), help=f"管线子集: {PIPES}")
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="合成时角色缩放")
    ap.add_argument("--pos", default="center")
    args = ap.parse_args()

    only = [p for p in args.only.split(",") if p]
    scene_args = ["--scene", str(args.scene)] if args.scene else []

    # 管线 B：背景（纯场景）
    if "bg" in only:
        run([sys.executable, os.path.join(BASE, "pipe_bg.py"),
             "--sb", args.sb, "--prefix", args.prefix] + scene_args)

    # 管线 A：角色（白底 + 透明）
    if "char" in only:
        run([sys.executable, os.path.join(BASE, "pipe_char.py"),
             "--sb", args.sb, "--character", args.character, "--prefix", args.prefix] + scene_args)

    # 管线 C：动画（白底 I2V + 逐帧抠图）
    if "anim" in only:
        run([sys.executable, os.path.join(BASE, "pipe_anim.py"),
             "--sb", args.sb, "--character", args.character, "--prefix", args.prefix] + scene_args)

    # 管线 D：音频（预配音）
    if "audio" in only:
        run([sys.executable, os.path.join(BASE, "pipe_audio.py"),
             "--sb", args.sb, "--prefix", args.prefix])

    # 管线 E：合成
    if "merge" in only:
        run([sys.executable, os.path.join(BASE, "merge.py"),
             "--sb", args.sb, "--prefix", args.prefix,
             "--scale", str(args.scale), "--pos", args.pos] + scene_args)


if __name__ == "__main__":
    main()
