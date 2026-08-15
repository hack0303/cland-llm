#!/usr/bin/env python3
"""管线 D：音频 — 预配音（克隆锚定尽力而为）+ 响度链（compose 侧）。

用法:
    python3 pipe_audio.py --sb storyboard.json --prefix lumo

输出: outputs_video/{prefix}/assets/audio/scene00X_voice.wav
"""
import argparse
import json
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "..", "i2v"))
from run_story import get_audio, healthy, TTS_URL, SFX_URL  # noqa: E402

OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sb", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=400)
    args = ap.parse_args()

    sb = json.load(open(args.sb))
    audio_dir = os.path.join(OUT_ROOT, args.prefix, "assets", "audio")
    os.makedirs(audio_dir, exist_ok=True)

    has_tts = healthy(TTS_URL)
    has_sfx = healthy(SFX_URL)
    print(f"[audio] TTS={'✅' if has_tts else '❌'} SFX={'✅' if has_sfx else '❌'}")

    if not has_tts:
        print("[audio] TTS 未运行，跳过配音")
        return

    print("[audio] 预配音（首句定音色 → 其余克隆同声纹）...")
    voice_ref = None
    for clip in sb["clips"]:
        s = clip["scene"]
        story_dir = os.path.join(OUT_ROOT, args.prefix)  # 兼容 get_audio 的 story_dir 结构
        # get_audio 期望 story_dir/audio/xxx.wav；这里用 assets/audio
        out = os.path.join(audio_dir, f"scene{s:03d}_voice.wav")
        if clip.get("voice") and not os.path.exists(out):
            wav = get_audio(story_dir, clip, "voice", clip["voice"], voice_ref=voice_ref)
            # get_audio 输出到 story_dir/audio/，移入 assets/audio
            src = os.path.join(story_dir, "audio", os.path.basename(wav))
            os.makedirs(audio_dir, exist_ok=True)
            os.rename(src, out)
            if voice_ref is None:
                voice_ref = (out, clip["voice"])
                print(f"  [audio] 音色锚定: {os.path.basename(out)}")
        elif os.path.exists(out):
            if voice_ref is None:
                voice_ref = (out, clip["voice"])
        else:
            continue
        if clip.get("sfx") and clip["sfx"] != "无" and has_sfx:
            sfx_out = os.path.join(audio_dir, f"scene{s:03d}_sfx.wav")
            if not os.path.exists(sfx_out):
                wav = get_audio(story_dir, clip, "sfx", clip["sfx"], args.seed + s)
                os.rename(os.path.join(story_dir, "audio", os.path.basename(wav)), sfx_out)
    print(f"[audio] 完成: {audio_dir}")


if __name__ == "__main__":
    main()
