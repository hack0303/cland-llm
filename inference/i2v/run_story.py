#!/usr/bin/env python3
"""分镜流水线执行器 — storyboard.json → 逐镜头生产 → 合成大视频。

依赖服务（未运行会自动跳过对应环节并提示）:
    10331 SDXL 出图 | 10337 ComfyUI I2V | 10333 TTS 配音 | 10336 SFX 音效

用法:
    python3 run_story.py --storyboard outputs_video/story001/storyboard.json
    python3 run_story.py --sb outputs_video/story001/storyboard.json --seed 42 --skip-audio

流程（每镜头）:
    1. SDXL 出参考图 → frames/sceneXXX.png
    2. AnimateDiff I2V   → clips/sceneXXX.mp4（512×512 @8fps）
    3. TTS 配音（voice 非空） → audio/sceneXXX_voice.wav
    4. SFX 音效（sfx 非"无"） → audio/sceneXXX_sfx.wav
    5. 回填 output 字段 + 全局时间戳 → storyboard.json
    6. compose 合成 → {prefix}_final.mp4
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
SDXL_URL = "http://127.0.0.1:10331"
TTS_URL = "http://127.0.0.1:10333"
SFX_URL = "http://127.0.0.1:10336"
OUT_ROOT = "/mnt/data/ai_workspace/outputs_video"
DEFAULT_NEGATIVE = ("blurry, low quality, distorted, watermark, text, wings, shoes, visible feet, deformed, "
                    "NSFW, suggestive, revealing, mature, sexy, sensual, tight clothing, cleavage, "
                    "pedestal, platform, base, stand, stage, podium, display stand, floor marker, shadow circle, "
                    "side view, profile, looking away, turned around, three-quarter view")
PROMPT_HUB_DIR = "/home/alice/work/agentic/alice-prompt-hub"
DEFAULT_LIGHTING = "soft warm glow in darkness"


def healthy(url, timeout=4):
    for path in ("health", "system_stats"):  # ComfyUI 用 /system_stats，无 /health
        try:
            with urllib.request.urlopen(f"{url}/{path}", timeout=timeout) as r:
                if r.status == 200:
                    return True
        except Exception:
            continue
    return False


def post_form(url, fields, files=None, timeout=600):
    """multipart 表单：fields=文本字段；files={name: (filename, bytes)} 文件字段"""
    import uuid
    boundary = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").encode()
    for name, (fname, fbytes) in (files or {}).items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; "
                 f"filename=\"{fname}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
        body += fbytes + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def post_json(url, data, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def render_shot_prompt(clip, style, lighting):
    """电影语言字段存在时用 prompt-hub story 渲染；否则旧拼接（向后兼容）"""
    has_film = any(k in clip for k in ("camera_move", "shot_size", "composition", "color", "aspect_ratio"))
    if not has_film:
        return f"{style}, {clip['image_prompt']}"
    cmd = ["uv", "run", "prompt-hub", "story",
           "--style", style, "--lighting", lighting or DEFAULT_LIGHTING,
           "--image-prompt", clip["image_prompt"], "--shot-no", str(clip["scene"]),
           "--camera-move", clip.get("camera_move", "FIXED"),
           "--shot-size", clip.get("shot_size", "MEDIUM_SHOT"),
           "--composition", clip.get("composition", "centered"),
           "--color", clip.get("color", "natural colors"),
           "--aspect-ratio", clip.get("aspect_ratio", "WIDESCREEN_169")]
    r = subprocess.run(cmd, cwd=PROMPT_HUB_DIR, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  [frame] ⚠️ prompt-hub story 渲染失败({r.stderr[-200:]})，回退旧拼接")
        return f"{style}, {clip['image_prompt']}"
    return r.stdout.strip()


def get_image(story_dir, clip, style, seed, lighting):
    out = os.path.join(story_dir, "frames", f"scene{clip['scene']:03d}.png")
    if os.path.exists(out):
        print(f"  [frame] 已存在，跳过: {out}")
        return out
    prompt = render_shot_prompt(clip, style, lighting)
    r = post_json(SDXL_URL + "/generate", {"prompt": prompt, "steps": 30, "seed": seed,
                                            "negative_prompt": DEFAULT_NEGATIVE})
    img = r.get("image") or r.get("path")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.rename(img, out) if os.path.exists(img) else subprocess.run(["cp", img, out])
    print(f"  [frame] {out}")
    return out


def get_clip(story_dir, clip, frame_path, seed, ref_image=None):
    out = os.path.join(story_dir, "clips", f"scene{clip['scene']:03d}.mp4")
    if os.path.exists(out):
        print(f"  [clip] 已存在，跳过: {out}")
        return out
    assert frame_path, "缺少参考图：SDXL 未运行或出图失败，I2V 必须有输入图"
    frames = max(8, int(round(clip["duration"] * 8)))  # 8fps × duration
    cmd = [sys.executable, os.path.join(BASE, "generate.py"),
           "--image", frame_path, "--prefix", f"scene{clip['scene']:03d}",
           "--frames", str(frames), "--seed", str(seed),
           "--outdir", os.path.join(story_dir, "clips"), "--size", "512x512"]
    if ref_image:
        cmd += ["--ref-image", ref_image]
    subprocess.run(cmd, check=True)
    print(f"  [clip] {out}")
    return out


def get_audio(story_dir, clip, kind, text, seed=None, voice_ref=None):
    """kind: voice|sfx；voice_ref=(wav_path, prompt_text) 时零样本克隆音色（音频一致性锚定）"""
    out = os.path.join(story_dir, "audio", f"scene{clip['scene']:03d}_{kind}.wav")
    if os.path.exists(out):
        print(f"  [{kind}] 已存在，跳过: {out}")
        return out
    if kind == "voice":
        fields = {"text": text, "gender": "female"}
        files = None
        if voice_ref:
            ref_wav, ref_text = voice_ref
            with open(ref_wav, "rb") as f:
                files = {"prompt_speech": (os.path.basename(ref_wav), f.read())}
            fields["prompt_text"] = ref_text
        r = post_form(TTS_URL + "/generate", fields, files=files)
    else:
        r = post_form(SFX_URL + "/generate", {"prompt": text, "seed": seed or 42})
    wav = r.get("wav") or r.get("path")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.rename(wav, out)
    print(f"  [{kind}] {out}")
    return out


def probe_duration(path):
    # ffprobe 不存在（imageio 单文件版 ffmpeg），用 ffmpeg -i 解析 stderr 的 Duration
    import re as _re
    r = subprocess.run(["ffmpeg", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
    m = _re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        raise RuntimeError(f"无法解析时长: {path}\n{r.stderr[-400:]}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def extract_last_frame(clip_path, story_dir):
    """从片段提取末帧（首帧接力：下镜头的 img2img 起点）"""
    out = os.path.join(story_dir, "frames", "chain_ref.png")
    subprocess.run(["ffmpeg", "-y", "-sseof", "-0.1", "-i", clip_path,
                    "-frames:v", "1", out], check=True, capture_output=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--storyboard", "--sb", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-audio", action="store_true", help="跳过配音/音效")
    ap.add_argument("--skip-video", action="store_true", help="跳过出图/I2V（仅合成）")
    ap.add_argument("--no-character-lock", action="store_true", help="关闭角色锚定（每镜头 IPAdapter 用各自首帧）")
    ap.add_argument("--chain-frames", action="store_true", help="首帧接力：镜头 N 首帧用镜头 N-1 末帧（同场景连续）")
    ap.add_argument("--character", default=None,
                    help="角色设定 JSON（char_sheet.py 产物）；锚定图=白底 front.png（透明 PNG 留给后期合成）")
    args = ap.parse_args()

    sb_path = args.storyboard
    with open(sb_path) as f:
        sb = json.load(f)
    story_dir = os.path.dirname(os.path.abspath(sb_path))
    prefix = os.path.basename(story_dir)
    style = sb["style"]
    clips = sb["clips"]

    has_sdxl = healthy(SDXL_URL)
    has_comfy = healthy("http://127.0.0.1:10337")
    has_tts = healthy(TTS_URL)
    has_sfx = healthy(SFX_URL)
    print(f"[run] 服务: SDXL={'✅' if has_sdxl else '❌'} ComfyUI={'✅' if has_comfy else '❌'} "
          f"TTS={'✅' if has_tts else '❌'} SFX={'✅' if has_sfx else '❌'}")
    if not has_comfy:
        print("[run] ComfyUI 未运行，无法出片", file=sys.stderr); sys.exit(1)

    # ── 0. 预配音阶段（配音纯文本→音频，不依赖画面，前置批量完成）──
    if not args.skip_audio and has_tts:
        print("[run] Phase 0: 预配音（首句定音色 → 其余克隆同声纹）...")
        voice_ref = None  # (基准 wav, 基准文本)：首句后设置，后续零样本克隆
        for clip in clips:
            s = clip["scene"]
            if clip.get("voice") and "voice" not in clip["output"]:
                wav = get_audio(story_dir, clip, "voice", clip["voice"], voice_ref=voice_ref)
                clip["output"]["voice"] = os.path.relpath(wav, story_dir)
                if voice_ref is None:
                    voice_ref = (wav, clip["voice"])
                    print(f"  [voice] 音色锚定: {os.path.basename(wav)}（后续台词克隆此声纹）")
            if clip.get("sfx") and clip["sfx"] != "无" and "sfx" not in clip["output"] and has_sfx:
                wav = get_audio(story_dir, clip, "sfx", clip["sfx"], args.seed + s)
                clip["output"]["sfx"] = os.path.relpath(wav, story_dir)
        print("[run] 预配音完成")

    # ── 1. 逐镜头生产 ──
    char_ref = None  # 角色锚定图（跨镜头锁角色）
    char_lighting = DEFAULT_LIGHTING
    if args.character and not args.no_character_lock:
        with open(args.character) as f:
            ch = json.load(f)
        char_ref = os.path.join(os.path.dirname(os.path.abspath(args.character)), ch["files"]["front"])
        char_lighting = ch.get("lighting", DEFAULT_LIGHTING)
        print(f"[run] 角色锚定: {char_ref} | lighting: {char_lighting}")
    lighting = sb.get("lighting") or char_lighting  # 分镜顶层可覆盖
    prev_frame = None  # 首帧接力（--chain-frames）
    for clip in clips:
        s = clip["scene"]
        print(f"[run] 镜头 {s}/{len(clips)} (duration={clip['duration']}s)")
        if not args.skip_video:
            if not has_sdxl:
                print("  [frame] 跳过：SDXL 未运行")
                frame = prev_frame if args.chain_frames else None
            else:
                frame = get_image(story_dir, clip, style, args.seed + s, lighting)
                clip["output"]["frame"] = os.path.relpath(frame, story_dir)
                # 镜头 1 出图即定妆照（后续镜头 IPAdapter 锚定它）
                if char_ref is None:
                    char_dir = os.path.join(story_dir, "assets")
                    os.makedirs(char_dir, exist_ok=True)
                    char_ref = os.path.join(char_dir, "character.png")
                    if not os.path.exists(char_ref):
                        subprocess.run(["cp", frame, char_ref])
                        print(f"  [char] 定妆照: {char_ref}")
            if args.chain_frames and prev_frame and frame is None:
                frame = prev_frame
            ref_for_clip = None if args.no_character_lock else (char_ref if char_ref else frame)
            clip_path = get_clip(story_dir, clip, frame, args.seed + s, ref_image=ref_for_clip)
            clip["output"]["clip"] = os.path.relpath(clip_path, story_dir)
            if args.chain_frames:
                prev_frame = extract_last_frame(clip_path, story_dir)
        if not args.skip_audio and has_tts and clip.get("voice") and "voice" not in clip["output"]:
            wav = get_audio(story_dir, clip, "voice", clip["voice"])
            clip["output"]["voice"] = os.path.relpath(wav, story_dir)
        if not args.skip_audio and has_sfx and clip.get("sfx") and clip["sfx"] != "无" and "sfx" not in clip["output"]:
            wav = get_audio(story_dir, clip, "sfx", clip["sfx"], args.seed + s)
            clip["output"]["sfx"] = os.path.relpath(wav, story_dir)

    # 回填 storyboard.json（output 字段）
    with open(sb_path, "w") as f:
        json.dump(sb, f, ensure_ascii=False, indent=2)

    # ── 2. 合成大视频（音频优先：镜头时长按语音实际长度动态撑开，声音零截断）──
    clip_files, voice_args, sfx_args, durations = [], [], [], []
    t_global = 0.0
    for clip in clips:
        clip_files.append(os.path.join(story_dir, clip["output"]["clip"]))
        v_len = 0.0
        if "voice" in clip["output"]:
            v_len = probe_duration(os.path.join(story_dir, clip["output"]["voice"]))
            voice_args += ["--voice", os.path.join(story_dir, clip["output"]["voice"]),
                           "--voice-at", str(round(t_global + clip["voice_at"], 2))]
        if "sfx" in clip["output"]:
            sfx_args += ["--sfx", os.path.join(story_dir, clip["output"]["sfx"]),
                         "--sfx-at", str(round(t_global + clip["sfx_at"], 2))]
        # 动态时长：镜头必须装得下完整语音（voice_at + 语音长度 + 0.3s 余量）
        target = max(clip["duration"], clip["voice_at"] + v_len + 0.3)
        if target > clip["duration"] + 0.05:
            print(f"[run] 镜头{clip['scene']}: 时长 {clip['duration']}s → {target:.1f}s（语音 {v_len:.1f}s，补帧）")
        durations.append(round(target, 2))
        t_global += target

    print(f"[run] 合成 {len(clip_files)} 个片段，总时长 {t_global:.1f}s")
    final = os.path.join(story_dir, f"{prefix}_final.mp4")
    subprocess.run([sys.executable, os.path.join(BASE, "compose.py"),
                    "--clips"] + clip_files + voice_args + sfx_args +
                   ["--durations"] + [str(d) for d in durations] +
                   ["--prefix", f"{prefix}_final", "--outdir", story_dir], check=True)
    print(f"[run] 完成: {final}")


if __name__ == "__main__":
    main()
