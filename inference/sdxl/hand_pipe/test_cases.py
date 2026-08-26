#!/usr/bin/env python3
"""工业管线「AI生图手脚崩坏」测试用例集 — 场景定义
每个场景 = 基线 vs 提示词方案 vs ControlNet 方案的对比样本
"""
import json
import os
import time
import urllib.request

API = os.environ.get("SDXL_API", "http://127.0.0.1:10331")

# 手部质量词（ShowDoc 方案1）
POS_HAND = ("perfect hands, detailed fingers, five fingers per hand, "
            "realistic palm, correct anatomy, detailed toes, correct limbs")
# 负面词（ShowDoc 方案1）
NEG_HAND = ("bad hands, missing fingers, extra fingers, deformed hands, "
            "merged fingers, mutated hands, bad feet, deformed toes, "
            "extra limbs, disfigured, distorted, blurry, low quality, watermark")

SCENES = [
    {
        "name": "s1_baseline_portrait",
        "prompt": "close-up portrait of a young woman, hands resting on her chin, studio lighting, photorealistic, 8k",
        "width": 1024, "height": 1024, "seed": 101, "steps": 30,
    },
    {
        "name": "s2_baseline_hand_gesture",
        "prompt": "a man showing a peace sign with his hand toward camera, shallow depth of field, photorealistic",
        "width": 1024, "height": 1024, "seed": 202, "steps": 30,
    },
    {
        "name": "s3_baseline_holding",
        "prompt": "a woman holding a coffee cup with both hands, warm cafe light, photorealistic close-up",
        "width": 1024, "height": 1024, "seed": 303, "steps": 30,
    },
    {
        "name": "s4_baseline_dancing",
        "prompt": "a ballet dancer mid-pose, arms extended, elegant hand positions, stage spotlight, photorealistic",
        "width": 1024, "height": 1024, "seed": 404, "steps": 30,
    },
]

SCENES_WITH_HAND_PROMPT = [
    {**s, "name": s["name"].replace("baseline", "promptfix"),
     "prompt": f"{s['prompt']}, {POS_HAND}"} for s in SCENES
]


def gen(name, prompt, neg, seed, steps=30, width=1024, height=1024):
    body = json.dumps({
        "prompt": prompt, "negative_prompt": neg, "seed": seed,
        "steps": steps, "width": width, "height": height,
    }).encode()
    req = urllib.request.Request(f"{API}/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def run_all(out_dir="/mnt/data/ai_workspace/outputs/hand_pipe"):
    os.makedirs(out_dir, exist_ok=True)
    results = []
    for s in SCENES:  # 基线（默认负面词）+ 提示词方案
        r = gen(s["name"], s["prompt"], NEG_HAND, s["seed"], s["steps"],
                s["width"], s["height"])
        results.append({"case": s["name"], **r})
        print(f"[{s['name']}] {r['image']} {r['seconds']}s", flush=True)
    for s in SCENES_WITH_HAND_PROMPT:
        r = gen(s["name"], s["prompt"], NEG_HAND, s["seed"], s["steps"],
                s["width"], s["height"])
        results.append({"case": s["name"], **r})
        print(f"[{s['name']}] {r['image']} {r['seconds']}s", flush=True)
    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDONE {len(results)} images -> {out_dir}/results.json")


if __name__ == "__main__":
    run_all()
