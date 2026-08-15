#!/usr/bin/env python3
"""大视频合成工具 — 多 I2V 片段拼接 + 配音/音效/BGM 混流。

用法:
    python3 compose.py --clips clip1.mp4 clip2.mp4 clip3.mp4 --prefix final
    # 声音（可多次指定，按出现顺序叠加）
    python3 compose.py --clips a.mp4 b.mp4 --voice voice.wav --voice-at 1.0 \
        --sfx boom.wav --sfx-at 4.5 --bgm bgm.mp3 --bgm-volume 0.3

流程:
    1. 片段统一转码（h264 / yuv420p / 8fps / 同分辨率）→ concat demuxer 拼接
    2. 音频轨按时间戳 adelay 放置 → amix 混音
    3. 视频 + 音频混流 → final.mp4
"""
import argparse
import os
import re
import subprocess
import tempfile

OUTPUT_DIR = "/mnt/data/ai_workspace/outputs_video"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg 失败: {' '.join(cmd[:6])}...\n{r.stderr[-800:]}")
    return r


def probe_duration(path):
    # ffprobe 不存在（imageio 单文件版 ffmpeg），用 ffmpeg -i 解析 stderr 的 Duration
    r = subprocess.run(["ffmpeg", "-i", path, "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        raise RuntimeError(f"无法解析时长: {path}\n{r.stderr[-400:]}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, help="片段 mp4 列表（按顺序）")
    ap.add_argument("--prefix", default="final", help="输出名前缀")
    ap.add_argument("--fps", type=float, default=8.0, help="统一帧率")
    ap.add_argument("--size", default="512x512", help="统一分辨率")
    ap.add_argument("--transition", type=float, default=None, help="片段间交叉淡化秒数（如 0.5；None=硬切）")
    ap.add_argument("--voice", action="append", default=[], help="配音 wav（可多次）")
    ap.add_argument("--voice-at", action="append", type=float, default=[], help="配音起始秒（对应 --voice 顺序）")
    ap.add_argument("--sfx", action="append", default=[], help="音效 wav（可多次）")
    ap.add_argument("--sfx-at", action="append", type=float, default=[], help="音效起始秒")
    ap.add_argument("--bgm", default=None, help="BGM 文件（自动循环铺底）")
    ap.add_argument("--bgm-volume", type=float, default=0.3)
    ap.add_argument("--outdir", default=OUTPUT_DIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    assert len(args.voice) == len(args.voice_at), "--voice 与 --voice-at 数量必须一致"
    assert len(args.sfx) == len(args.sfx_at), "--sfx 与 --sfx-at 数量必须一致"

    with tempfile.TemporaryDirectory() as td:
        # ── 1. 统一转码每个片段（concat 要求同编码/分辨率/帧率）──
        norm_clips = []
        for i, clip in enumerate(args.clips):
            nc = os.path.join(td, f"n{i:03d}.mp4")
            run(["ffmpeg", "-y", "-i", clip, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                 "-r", str(args.fps), "-s", args.size, "-an", nc])
            norm_clips.append(nc)

        # ── 2. 拼接（硬切 concat / 淡化 xfade）──
        if args.transition:
            # xfade 链：每段时长递减一个 transition
            durs = [probe_duration(c) for c in norm_clips]
            offsets = []
            acc = durs[0]
            for d in durs[1:]:
                offsets.append(acc - args.transition)
                acc += d - args.transition
            fc = []
            for i in range(len(norm_clips) - 1):
                fc.append(f"[{i}:v][{i+1}:v]xfade=transition=fade:duration={args.transition}:offset={offsets[i]:.3f}[v{i+1}]")
            fc.append("[v%d]" % (len(norm_clips) - 1) + "format=yuv420p[vout]")
            vconcat = os.path.join(td, "vconcat.mp4")
            run(["ffmpeg", "-y"] + sum([["-i", c] for c in norm_clips], []) +
                ["-filter_complex", ";".join(fc), "-map", "[vout]", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-r", str(args.fps), vconcat])
            total = acc
        else:
            lst = os.path.join(td, "list.txt")
            with open(lst, "w") as f:
                for c in norm_clips:
                    f.write(f"file '{c}'\n")
            vconcat = os.path.join(td, "vconcat.mp4")
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", vconcat])
            total = sum(probe_duration(c) for c in norm_clips)

        print(f"[compose] 视频拼接完成: {total:.1f}s")

        # ── 3. 音频轨：adelay 对齐 + 防重叠截断 + amix ──
        # 先收集全部音频时间戳（全局秒）
        audio_events = []  # (全局起点, 类型, 文件)
        for wav, at in zip(args.voice + args.sfx, args.voice_at + args.sfx_at):
            audio_events.append((at, "voice" if wav in args.voice else "sfx", wav))
        if args.bgm:
            audio_events.append((0.0, "bgm", args.bgm))
        audio_events.sort()
        # 每段音频截断到与下一段间隔（防混叠；voice 保底 1s）
        cut_ends = {}
        for i, (at, typ, wav) in enumerate(audio_events):
            nxt = audio_events[i + 1][0] if i + 1 < len(audio_events) else total
            limit = max(1.0, nxt - at - 0.2)
            cut_ends[(at, typ, wav)] = limit

        audio_srcs, labels = [], []
        aidx = 0
        # 输入 0 = vconcat（纯视频），音频从输入 1 开始 → [aidx+1:a]
        for wav, at in zip(args.voice + args.sfx, args.voice_at + args.sfx_at):
            ms = int(at * 1000)
            limit = cut_ends[(at, "voice" if wav in args.voice else "sfx", wav)]
            # 顺序关键：先 atrim 截原音频，再 adelay 延迟（反序会把延迟静音截掉，语音全丢）
            labels.append(f"[{aidx+1}:a]atrim=0:{limit:.2f},adelay={ms}|{ms}[a{aidx}]")
            audio_srcs.append(wav)
            aidx += 1
        if args.bgm:
            labels.append(f"[{aidx+1}:a]volume={args.bgm_volume},aloop=loop=-1:size=2e9,atrim=0:{total:.2f}[a{aidx}]")
            audio_srcs.append(args.bgm)
            aidx += 1

        final = os.path.join(args.outdir, f"{args.prefix}.mp4")
        if audio_srcs:
            mixins = "".join(f"[a{i}]" for i in range(aidx))
            labels.append(f"{mixins}amix=inputs={aidx}:normalize=0[aout]")
            fc = labels
            amap = "[aout]"
            run(["ffmpeg", "-y", "-i", vconcat] + sum([["-i", s] for s in audio_srcs], []) +
                ["-filter_complex", ";".join(fc), "-map", "0:v", "-map", amap,
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", final])
        else:
            run(["ffmpeg", "-y", "-i", vconcat, "-c", "copy", final])

        print(f"[compose] 输出: {final}")


if __name__ == "__main__":
    main()
