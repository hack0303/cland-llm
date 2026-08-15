#!/usr/bin/env python3
"""SDXL 常驻 API 服务 (Tesla P40 / FP16) — 模型常驻显存，即调即出"""
import argparse
import time

import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

MODEL_DIR = "/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0"
app = FastAPI(title="SDXL Image Service (INT8)")


class GenRequest(BaseModel):
    prompt: str = Field(..., description="提示词")
    negative_prompt: str = "blurry, low quality, distorted, watermark, deformed, bad anatomy, extra limbs, poorly drawn hands, text, jpeg artifacts, ugly, duplicate, oversaturated, extra fingers"
    steps: int = 30
    width: int = 1024
    height: int = 1024
    seed: int = 42
    guidance_scale: float = 7.5


@app.on_event("startup")
def load_model():
    global pipe
    t0 = time.time()
    print("[*] Loading SDXL (INT8, first load ~10min)...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float16, load_in_8bit=True, use_safetensors=True
    )
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    print(f"[*] Model loaded in {time.time()-t0:.0f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "stable-diffusion-xl-base-1.0"}


@app.post("/generate")
def generate(req: GenRequest):
    g = torch.Generator("cpu").manual_seed(req.seed)  # 8bit 模式下 VAE 在 CPU，需 cpu generator
    t0 = time.time()
    img = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        num_inference_steps=req.steps,
        width=req.width,
        height=req.height,
        guidance_scale=req.guidance_scale,
        generator=g,
    ).images[0]
    dt = time.time() - t0
    out_path = f"/mnt/data/ai_workspace/outputs/sdxl_{int(t0)}_{req.seed}.png"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path)
    return {
        "image": out_path,
        "seconds": round(dt, 1),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "seed": req.seed,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10331)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
