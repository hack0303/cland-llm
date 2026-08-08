#!/usr/bin/env python3
"""SDXL 文本生图脚本 (Tesla P40 / FP16)"""
import argparse
import time

import torch
from diffusers import StableDiffusionXLPipeline

MODEL_DIR = "/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="A red fox in a snowy pine forest at golden hour, photorealistic, sharp focus, soft bokeh")
    ap.add_argument("--negative", type=str, default="blurry, low quality, distorted, watermark")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=str, default="sdxl_output.png")
    args = ap.parse_args()

    print(f"[*] Loading SDXL from {MODEL_DIR} (FP16)...")
    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = pipe.to("cuda")
    # P40 (Pascal) 老卡稳妥配置
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    print(f"[*] Model loaded in {time.time()-t0:.1f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    g = torch.Generator("cuda").manual_seed(args.seed)
    print(f"[*] Generating {args.width}x{args.height}, {args.steps} steps...")
    t1 = time.time()
    img = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        width=args.width,
        height=args.height,
        guidance_scale=7.5,
        generator=g,
    ).images[0]
    dt = time.time() - t1
    img.save(args.output)
    print(f"[*] Saved -> {args.output}  ({dt:.1f}s, "
          f"peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
