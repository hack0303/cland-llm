#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PaddleOCR 服务（C-Land 本机，端口 10337）
环境: ocr_env (paddlepaddle-gpu 2.6.1 cu118 + paddleocr 2.7.3)
注意: paddle 2.6 需要 libcudnn.so（无版本号），已软链至 paddle/libs/
启动: nohup .../ocr_env/bin/python server.py > /tmp/ocr_server.log 2>&1 &
"""
import base64
import io
import time
import uuid

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
from PIL import Image

# --- 1. 初始化模型（进程内单例，首次 ~30s） ---
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=True, show_log=False)

app = FastAPI(title="C-Land OCR (PaddleOCR)", version="1.0.0")

# 输出目录
OUT_DIR = "/mnt/data/ai_workspace/outputs_ocr"


def _predict(img: Image.Image) -> dict:
    t0 = time.time()
    # 转 numpy（paddleocr 2.7 接受 ndarray）
    arr = np.array(img.convert("RGB"))
    result = ocr.ocr(arr, cls=True)
    lines = []
    for line in (result[0] or []):
        box, (text, score) = line
        lines.append({
            "text": text,
            "score": round(float(score), 4),
            "box": [[int(round(x)), int(round(y))] for x, y in box],
        })
    return {"lines": lines, "elapsed": round(time.time() - t0, 3)}


@app.get("/health")
def health():
    return {"status": "ok", "service": "paddleocr", "gpu": True}


@app.post("/ocr")
async def ocr_upload(image: UploadFile = File(...)):
    data = await image.read()
    img = Image.open(io.BytesIO(data))
    result = _predict(img)
    return JSONResponse({"request_id": uuid.uuid4().hex[:12], **result})


@app.post("/ocr/base64")
async def ocr_base64(body: dict):
    img = Image.open(io.BytesIO(base64.b64decode(body["image"])))
    result = _predict(img)
    return JSONResponse({"request_id": uuid.uuid4().hex[:12], **result})


if __name__ == "__main__":
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10337)
