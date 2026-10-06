#!/usr/bin/env python3
"""SDXL 常驻 API 服务 (Tesla P40 / FP16) — 模型常驻显存，即调即出

加载说明（#323 C 修复）：
- 用 **fp16 权重目录**，`torch_dtype=float16`，**不再传 `load_in_8bit`**（该参数会被 diffusers 忽略 →
  实为全精度加载 → 宿主 RAM 压力大，本机 15GB RAM 下加载被杀）。
- fp16 权重 ~6.5G（UNET fp16 ≈ fp32 的一半）→ 宿主 RAM 需求近乎减半。
"""
import argparse
import os
import threading
import time

import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

MODEL_DIR = os.environ.get("SDXL_MODEL_DIR",
                           "/mnt/data/ai_workspace/models/stable-diffusion-xl-base-1.0-fp16")
DEVICE = os.environ.get("SDXL_DEVICE", "cuda")
app = FastAPI(title="SDXL Image Service (FP16)")

# diffusers 调度器非线程安全（并发请求共享 step_index 会越界 500）→ 全局锁串行化
GEN_LOCK = threading.Lock()


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
    model_dir = os.environ.get("SDXL_MODEL_DIR", MODEL_DIR)
    print(f"[*] Loading SDXL (FP16) from {model_dir} ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir, torch_dtype=torch.float16, use_safetensors=True,
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    print(f"[*] Model loaded in {time.time()-t0:.0f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "stable-diffusion-xl-base-1.0"}


@app.post("/generate")
def generate(req: GenRequest):
    g = torch.Generator("cpu").manual_seed(req.seed)  # CPU generator：可复现且不占 GPU 状态
    t0 = time.time()
    with GEN_LOCK:  # 串行化：P40 单卡并发无收益，防调度器状态竞争
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
    ap.add_argument("--model-dir", default=MODEL_DIR)
    args = ap.parse_args()
    os.environ["SDXL_MODEL_DIR"] = args.model_dir
    uvicorn.run(app, host=args.host, port=args.port)
