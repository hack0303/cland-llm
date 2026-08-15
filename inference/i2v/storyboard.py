#!/usr/bin/env python3
"""分镜脚本生成工具 — 故事文本 → 镜头 JSON（LLM: 本机 Gemma-4 10303）。

用法:
    python3 storyboard.py --story "一只猫宇航员在月球上探险..." --prefix story001
    python3 storyboard.py --story-file story.txt --prefix story001 --clips 6

输出: /mnt/data/ai_workspace/outputs_video/story001.json
后续: 分镜 JSON 可直接驱动流水线（SDXL 出图 → I2V → TTS/SFX → compose 合成）
"""
import argparse
import json
import os
import sys
import time
import urllib.request

OUTPUT_DIR = "/mnt/data/ai_workspace/outputs_video"
LLM_URL = "http://127.0.0.1:10303/v1/chat/completions"
LLM_MODEL = "gemma-4-26b"

SYSTEM_PROMPT = """你是游戏短片分镜导演。根据用户提供的故事，输出分镜脚本 JSON（严格 JSON，不要任何其他文字）。

JSON 结构：
{
  "title": "标题（中文）",
  "style": "整体画风（英文，如 cute cartoon, soft lighting）",
  "clips": [
    {
      "scene": 1,
      "image_prompt": "画面提示词（英文）",
      "motion_prompt": "运动提示词（英文）",
      "voice": "旁白/台词（中文，该镜头说的话）",
      "voice_at": 0.0,
      "sfx": "音效描述（中文，如 风声、脚步声）",
      "sfx_at": 0.0,
      "duration": 2
    }
  ]
}

提示词书写要求（必须遵守）：
- image_prompt 英文四段式：{主体} {姿态}, {环境}, {光照}, {风格+质量词}，15~30 词一句；
  **角色描述短语跨镜头固定复用**（同一角色各镜头用一模一样的开头，只换动作/环境）；禁止否定词
- motion_prompt 只写轻微运动（AnimateDiff 能力边界）：subtle motion, gentle breeze, soft breathing,
  slight waving, blinking, leaves drifting 等；**禁止 run/jump/somersault/fighting/explosion/fast pan 等大幅度运动词**；统一 subtle/gentle/soft 开头
- voice 一句一镜 ≤30 字口语化；sfx ≤10 字短描述，无音效写"无"
- 相邻镜头场景词连贯（同一地点用相同英文表达）

其他规则：
- clips 数量 = 故事自然分镜数（通常 3~8 个），每个镜头 duration 默认 2 秒
- voice 是镜头开始时朗读的台词，voice_at 用镜头内偏移秒数（须小于 duration）
- 只输出 JSON 本身，不要 markdown 代码块、不要解释"""

SCHEMA_KEYS = ["scene", "image_prompt", "motion_prompt", "voice", "voice_at", "sfx", "sfx_at", "duration"]


def call_llm(story: str, max_tokens: int = 4096, retries: int = 2) -> dict:
    body = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": story},
        ],
        "temperature": 0.6,
        "max_tokens": max_tokens,
    }
    for attempt in range(retries + 1):
        req = urllib.request.Request(LLM_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.loads(r.read())
            content = resp["choices"][0]["message"]["content"]
            # 去掉可能的 markdown 代码块包裹
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            sb = json.loads(content)
            validate(sb)
            # 规范 v1.0：自动补齐 schema 元信息
            sb["schema_version"] = "1.0"
            sb["total_duration"] = round(sum(c["duration"] for c in sb["clips"]), 2)
            for i, c in enumerate(sb["clips"]):
                c["output"] = {}
            return sb
        except Exception as e:
            print(f"[storyboard] 第 {attempt+1} 次尝试失败: {type(e).__name__}: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(3)
    sys.exit(1)


def validate(sb: dict):
    assert isinstance(sb.get("clips"), list) and len(sb["clips"]) > 0, "clips 为空"
    assert isinstance(sb.get("title"), str) and sb["title"], "缺少 title"
    assert isinstance(sb.get("style"), str) and sb["style"], "缺少 style"
    for i, c in enumerate(sb["clips"]):
        for k in SCHEMA_KEYS:
            assert k in c, f"clips[{i}] 缺少字段 {k}"
        assert c["scene"] == i + 1, f"clips[{i}].scene 须为 {i+1}"
        assert isinstance(c["duration"], (int, float)) and 1 <= c["duration"] <= 6, "duration 须 1~6s"
        assert c["voice_at"] < c["duration"] and c["sfx_at"] < c["duration"], "音频偏移超出镜头时长"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", help="故事文本")
    ap.add_argument("--story-file", help="故事文件路径")
    ap.add_argument("--prefix", default="story001")
    ap.add_argument("--clips", type=int, default=0, help=">0 时要求 LLM 精确生成 N 个镜头")
    args = ap.parse_args()
    assert args.story or args.story_file, "需要 --story 或 --story-file"

    story = args.story or open(args.story_file).read()
    if args.clips > 0:
        story = f"{story}\n\n（要求：精确生成 {args.clips} 个镜头）"

    print(f"[storyboard] 调用 LLM 生成分镜...（{LLM_URL}）")
    t0 = time.time()
    sb = call_llm(story)
    out_dir = os.path.join(OUTPUT_DIR, args.prefix)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "storyboard.json")
    with open(out, "w") as f:
        json.dump(sb, f, ensure_ascii=False, indent=2)

    print(f"[storyboard] 完成（{time.time()-t0:.0f}s），{len(sb['clips'])} 个镜头 → {out}")
    for c in sb["clips"]:
        print(f"  镜头{c['scene']}: {c['duration']}s | {c['image_prompt'][:60]}...")
        if c["voice"]:
            print(f"          配音: {c['voice'][:50]}")


if __name__ == "__main__":
    main()
