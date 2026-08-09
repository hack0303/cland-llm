#!/usr/bin/env python3
"""SenseVoice 语音识别服务 (GPU 1 / 端口 10334)"""
import argparse
import os
import time

import torch
from fastapi import FastAPI, File, Form, UploadFile
import uvicorn

MODEL_DIR = "/mnt/data/ai_workspace/models/SenseVoiceSmall"
OUT_DIR = "/mnt/data/ai_workspace/outputs_audio"
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="SenseVoice ASR Service")


@app.on_event("startup")
def load_model():
    global model
    from funasr import AutoModel
    t0 = time.time()
    print("[*] Loading SenseVoiceSmall ...", flush=True)
    model = AutoModel(model=MODEL_DIR, device="cuda:0", disable_update=True, hub="hf")
    print(f"[*] Loaded in {time.time()-t0:.0f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "SenseVoiceSmall"}


@app.post("/recognize")
def recognize(
    audio: UploadFile = File(...),
    language: str = Form("auto"),
):
    """audio → text + 情感/事件标签（<|zh|><|HAPPY|><|Speech|>...）"""
    t0 = time.time()
    tmp = f"/tmp/sense_audio_{int(t0)}.wav"
    with open(tmp, "wb") as f:
        f.write(audio.file.read())
    try:
        res = model.generate(input=tmp, language=language, use_itn=True)
        raw = res[0]["text"]
        os.remove(tmp)
        import re
        tags = re.findall(r"<\|[^|]+\|>", raw)
        text = re.sub(r"<\|[^|]+\|>", "", raw).strip()
        return {
            "model": "SenseVoice",
            "text": text,
            "tags": tags,
            "language": next((t[2:-2] for t in tags if t[2:-2] in ("zh", "en", "yue", "ja", "ko")), None),
            "emotion": next((t[2:-2] for t in tags if t[2:-2] in ("HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED")), None),
            "seconds": round(time.time() - t0, 1),
        }
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10334)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
