#!/usr/bin/env python3
"""LoRA 手部增强对比测试（单 scale 一次）
用法: python3 test_lora.py --scale 0     # 无 LoRA 基线
      python3 test_lora.py --scale 0.6   # nice-hands 权重 0.6
评估: DWPose 手部 21 点完整度 + 置信度
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dwpose import DWPose, LEFT_HAND, RIGHT_HAND

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

BASE = "/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0"
LORAS = {
    "none": None,
    "nice_hands": "/mnt/data/ai_workspace/models/sdxl_loras/nice_hands.safetensors",
    "goodhands": "/mnt/data/ai_workspace/models/sdxl_loras/GoodHands-beta2.safetensors",
    "perfect_hands_v2": "/mnt/data/ai_workspace/models/sdxl_loras/perfect_hands_v2.safetensors",
    "muapi": "/mnt/data/ai_workspace/models/sdxl_loras/muapi_all_in_one.safetensors",
}


def load_kohya_lora_manual(pipe, lora_path, scale):
    """手动注入 kohya LoRA（绕开 diffusers 0.39 的 rank 推断 bug）
    将 lora_linear_layer down/up 权重按 delta = alpha/rank * up@down 加到模块参数
    """
    from safetensors import safe_open
    from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_lora_to_diffusers
    sf = safe_open(lora_path, framework="pt")
    sd = {k: sf.get_tensor(k) for k in sf.keys()}
    sd2, alphas = _convert_non_diffusers_lora_to_diffusers(sd)
    components = {"text_encoder": pipe.text_encoder,
                  "text_encoder_2": pipe.text_encoder_2,
                  "unet": pipe.unet}
    n_lin = n_conv = 0
    for key, down in sd2.items():
        if not key.endswith(".lora_linear_layer.down.weight"):
            continue
        up_key = key.replace(".down.weight", ".up.weight")
        base = key.replace(".lora_linear_layer.down.weight", "")
        alpha = float(alphas.get(key + ".alpha", 1.0))
        parts = base.split(".")
        comp = parts[0]
        if comp not in components:
            continue
        mod = components[comp]
        try:
            for p in parts[1:]:
                mod = getattr(mod, p)
        except AttributeError:
            continue
        up = sd2[up_key]
        rank = down.shape[0]
        delta = (alpha / rank) * (up @ down)
        delta = delta.to(mod.weight.dtype)
        if mod.weight.ndim == 4:  # Conv2d
            delta = delta.reshape(mod.weight.shape)
            n_conv += 1
        else:
            n_lin += 1
        with torch.no_grad():
            mod.weight.add_(scale * delta)
    print(f"[*] manual LoRA injected: {n_lin} linear + {n_conv} conv (scale={scale})", flush=True)
OUT = "/mnt/data/ai_workspace/outputs/hand_pipe/lora_test"
os.makedirs(OUT, exist_ok=True)

NEG = ("bad hands, missing fingers, extra fingers, deformed hands, merged fingers, "
       "mutated hands, bad feet, deformed toes, extra limbs, disfigured, distorted, "
       "blurry, low quality, watermark, bad anatomy")

PROMPTS = [
    ("peace_sign", "a man showing a peace sign with his hand toward camera, shallow depth of field, photorealistic"),
    ("holding_cup", "a woman holding a coffee cup with both hands, warm cafe light, photorealistic close-up"),
    ("chin_hands", "close-up portrait of a young woman, hands resting on her chin, studio lighting, photorealistic"),
    ("dancer", "a ballet dancer mid-pose, arms extended, elegant hand positions, stage spotlight, photorealistic"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora", default="nice_hands",
                    choices=list(LORAS.keys()))
    a = ap.parse_args()

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, torch_dtype=torch.float16, use_safetensors=True)
    lora_path = LORAS[a.lora]
    if lora_path and a.scale > 0:
        if a.lora == "perfect_hands_v2":
            # 手动注入（diffusers 0.39 对该文件 rank 推断崩溃）
            load_kohya_lora_manual(pipe, lora_path, a.scale)
        else:
            print(f"[*] loading LoRA {a.lora} scale={a.scale}", flush=True)
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=a.scale)
    pipe = pipe.to("cuda:0")
    pipe.enable_attention_slicing()

    dp = DWPose()
    for name, prompt in PROMPTS:
        g = torch.Generator("cpu").manual_seed(a.seed)
        img = pipe(prompt=prompt, negative_prompt=NEG, num_inference_steps=30,
                   width=1024, height=1024, guidance_scale=7.5, generator=g).images[0]
        tag = f"{a.lora}_{str(a.scale).replace('.', 'p')}"
        p = f"{OUT}/{name}_{tag}_{a.seed}.png"
        img.save(p)
        # 评估
        bgr = cv2.imread(p)
        dets = dp.detect_full(bgr)
        best = max(dets, key=lambda d: int((d["kps"][LEFT_HAND, 2] > 0.3).sum()) +
                   int((d["kps"][RIGHT_HAND, 2] > 0.3).sum())) if dets else None
        if best:
            kps = best["kps"]
            hl = int((kps[LEFT_HAND, 2] > 0.3).sum())
            hr = int((kps[RIGHT_HAND, 2] > 0.3).sum())
            conf = max(float(kps[LEFT_HAND, 2].max()), float(kps[RIGHT_HAND, 2].max()))
            print(f"  {name}: handL={hl:2d} handR={hr:2d} conf={conf:.2f}", flush=True)
        else:
            print(f"  {name}: NO PERSON DETECTED", flush=True)
        del g
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
