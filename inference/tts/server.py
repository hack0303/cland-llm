#!/usr/bin/env python3
"""Spark-TTS 文本转语音服务 (GPU 1 / 端口 10333)"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile
import uvicorn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "spark_tts_src"))
from cli.SparkTTS import SparkTTS  # noqa: E402

MODEL_DIR = "/mnt/data/ai_workspace/models/Spark-TTS-0.5B"
OUT_DIR = "/mnt/data/ai_workspace/outputs_audio"
os.makedirs(OUT_DIR, exist_ok=True)

torch.backends.cudnn.enabled = False  # P40: cuDNN 9.x 不支持 sm_61

app = FastAPI(title="Spark-TTS Service")


@app.on_event("startup")
def load_model():
    global model
    t0 = time.time()
    print("[*] Loading Spark-TTS ...", flush=True)
    model = SparkTTS(MODEL_DIR, torch.device("cuda:0"))
    print(f"[*] Loaded in {time.time()-t0:.0f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "Spark-TTS-0.5B"}


@app.post("/generate")
def generate(
    text: str = Form(...),
    gender: str = Form("female"),
    pitch: str = Form("moderate"),
    speed: str = Form("moderate"),
    prompt_speech: UploadFile = File(None),
    prompt_text: str = Form(None),
):
    """text → wav。无 prompt 时按 gender/pitch/speed 控制；有 prompt 时零样本克隆音色。"""
    t0 = time.time()
    prompt_path = None
    if prompt_speech is not None:
        prompt_path = f"/tmp/spark_prompt_{int(t0)}.wav"
        with open(prompt_path, "wb") as f:
            f.write(prompt_speech.file.read())
    with torch.no_grad():
        wav = model.inference(
            text, prompt_path, prompt_text=prompt_text,
            gender=gender, pitch=pitch, speed=speed,
        )
    if prompt_path:
        os.remove(prompt_path)
    out_path = f"{OUT_DIR}/tts_{int(t0)}.wav"
    sf.write(out_path, wav, samplerate=16000)
    return {
        "model": "Spark-TTS",
        "wav": out_path,
        "seconds": round(time.time() - t0, 1),
        "text": text,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10333)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
