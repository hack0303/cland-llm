#!/usr/bin/env python3
"""AudioGen 音效生成服务 (GPU 1 / 端口 10336)"""
import argparse
import os
import time

import torch
import torchaudio
from fastapi import FastAPI, Form
import uvicorn

OUT_DIR = "/mnt/data/ai_workspace/outputs_audio"
os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cudnn.enabled = False  # P40: cuDNN 9.x 不支持 sm_61

app = FastAPI(title="AudioGen SFX Service")


@app.on_event("startup")
def load_model():
    global model
    from audiocraft.models import AudioGen
    t0 = time.time()
    print("[*] Loading AudioGen-medium ...", flush=True)
    model = AudioGen.get_pretrained("facebook/audiogen-medium", device="cuda")
    print(f"[*] Loaded in {time.time()-t0:.0f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "AudioGen-medium"}


@app.post("/generate")
def generate(
    prompt: str = Form(...),
    duration: float = Form(5.0),
    seed: int = Form(42),
):
    """prompt → 音效 wav（英文提示词效果最佳，如 'dog barking in a park'）"""
    t0 = time.time()
    torch.manual_seed(seed)
    model.set_generation_params(duration=duration)
    wav = model.generate([prompt])
    out_path = f"{OUT_DIR}/sfx_{int(t0)}_{seed}.wav"
    torchaudio.save(out_path, wav[0].cpu(), model.sample_rate)
    return {
        "model": "AudioGen",
        "wav": out_path,
        "seconds": round(time.time() - t0, 1),
        "duration": duration,
        "sample_rate": model.sample_rate,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10336)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
