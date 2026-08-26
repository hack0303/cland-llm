#!/usr/bin/env python3
"""AI 生图手脚崩坏修复 — 工业管线常驻服务 (GPU1 :10335)
环节：txt2img(SDXL) / DWPose 姿态 / ControlNet(OpenPose·Depth·Canny) /
      inpaint 局部重绘(SDXL-inpaint) / Real-ESRGAN 超分
"""
import argparse
import os
import threading
import time

import cv2
import numpy as np
import torch
from diffusers import (PipelineQuantizationConfig, StableDiffusionXLPipeline, StableDiffusionXLControlNetPipeline,
                       StableDiffusionXLInpaintPipeline, ControlNetModel,
                       MultiControlNetModel, AutoencoderKL)
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field
import uvicorn

from dwpose import DWPose
from rrdbnet import RealESRGANUpscaler

BASE_DIR = "/mnt/data/ai_workspace/models"
CN_DIR = f"{BASE_DIR}/sdxl_controlnet"
OUT_DIR = "/mnt/data/ai_workspace/outputs/hand_pipe"

# 8bit 量化配置（diffusers 0.39 PipelineQuantizationConfig API）
Q8 = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True},
    components_to_quantize=["unet", "controlnet", "text_encoder", "text_encoder_2"],
)
Q8_INP = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_8bit",
    quant_kwargs={"load_in_8bit": True},
    components_to_quantize=["unet", "text_encoder", "text_encoder_2"],
)

NEG = ("bad hands, missing fingers, extra fingers, deformed hands, merged fingers, "
       "mutated hands, bad feet, deformed toes, extra limbs, disfigured, distorted, "
       "blurry, low quality, watermark, bad anatomy")

app = FastAPI(title="Hand-Fix Industrial Pipeline (SDXL+ControlNet+Inpaint+ESRGAN)")
GEN_LOCK = threading.Lock()
PIPE = {}  # 组件容器


class GenReq(BaseModel):
    prompt: str
    negative_prompt: str = NEG
    steps: int = 30
    width: int = 1024
    height: int = 1024
    seed: int = 42
    guidance_scale: float = 7.5


class CNReq(BaseModel):
    prompt: str
    negative_prompt: str = NEG
    condition: str = Field("openpose", description="openpose|depth|canny")
    image: str = Field("", description="条件图路径；空则自动 openpose 检测")
    source_image: str = Field("", description="从该图自动生成条件图")
    steps: int = 30
    width: int = 1024
    height: int = 1024
    seed: int = 42
    guidance_scale: float = 7.5
    cn_strength: float = 0.75


class InpaintReq(BaseModel):
    image: str
    prompt: str = "perfect detailed hand, five fingers, correct anatomy"
    negative_prompt: str = NEG
    hand_box: str = Field("", description="x1,y1,x2,y2; 空则自动检测")
    denoise: float = Field(0.45, description="重绘强度 0.3-0.5 推荐")
    steps: int = 30
    seed: int = 42


class UpscaleReq(BaseModel):
    image: str
    model: str = "ultrasharp"  # ultrasharp | esrgan


class PipeReq(BaseModel):
    prompt: str
    negative_prompt: str = NEG
    steps: int = 30
    seed: int = 42
    width: int = 1024
    height: int = 1024
    guidance_scale: float = 7.5
    use_cn: bool = True
    use_inpaint: bool = True
    use_upscale: bool = True
    cn_strength: float = 0.75
    inpaint_denoise: float = 0.45


# ---------------- 模型加载 ----------------
def load_cn(name):
    """懒加载 ControlNet：重建 cn_pipe（单 controlnet），先释放旧管线显存"""
    print(f"[*] Loading ControlNet {name} (fp16, lazy) ...", flush=True)
    cn = ControlNetModel.from_pretrained(f"{CN_DIR}/{name}",
                                         torch_dtype=torch.float16,
                                         use_safetensors=True)
    cn = cn.to("cuda:1")
    mc = MultiControlNetModel([cn])
    if "cn_pipe" in PIPE:
        old_pipe = PIPE.pop("cn_pipe")
        for mod in (old_pipe.unet, old_pipe.text_encoder, old_pipe.text_encoder_2,
                    old_pipe.vae, *getattr(old_pipe, "controlnet", [])):
            if mod is not None:
                mod.to("cpu")
        del old_pipe
        torch.cuda.empty_cache()
    cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        f"{BASE_DIR}/stable-diffusion-xl-base-1.0", controlnet=mc,
        torch_dtype=torch.float16,
        quantization_config=Q8, use_safetensors=True)
    cn_pipe = cn_pipe.to("cuda:1")
    cn_pipe.enable_attention_slicing()
    PIPE["cn_pipe"] = cn_pipe
    PIPE["cn_names"] = [name]
    return cn_pipe


@app.on_event("startup")
def load_all():
    t0 = time.time()
    torch.cuda.set_device(1)  # GPU1
    print("[*] Loading SDXL base (8bit) ...", flush=True)
    base = StableDiffusionXLPipeline.from_pretrained(
        f"{BASE_DIR}/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
        quantization_config=Q8, use_safetensors=True)
    base = base.to("cuda:1")
    base.enable_attention_slicing()
    base.enable_vae_slicing()
    PIPE["base"] = base

    print("[*] Loading ControlNet openpose (fp16) ...", flush=True)
    cn = ControlNetModel.from_pretrained(f"{CN_DIR}/openpose",
                                         torch_dtype=torch.float16,
                                         use_safetensors=True)
    cn = cn.to("cuda:1")
    mc = MultiControlNetModel([cn])  # 显式包装，绕过 from_pretrained 的 list 处理
    cn_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        f"{BASE_DIR}/stable-diffusion-xl-base-1.0", controlnet=mc,
        torch_dtype=torch.float16,
        quantization_config=Q8, use_safetensors=True)
    cn_pipe = cn_pipe.to("cuda:1")
    cn_pipe.enable_attention_slicing()
    PIPE["cn_pipe"] = cn_pipe
    PIPE["cn_names"] = ["openpose"]  # depth/canny 懒加载

    print("[*] Loading SDXL-inpaint (8bit) ...", flush=True)
    inp = StableDiffusionXLInpaintPipeline.from_pretrained(
        f"{BASE_DIR}/sdxl-inpaint", torch_dtype=torch.float16,
        quantization_config=Q8_INP, use_safetensors=True)
    inp = inp.to("cuda:1")
    inp.enable_attention_slicing()
    inp.enable_vae_slicing()
    PIPE["inpaint"] = inp

    print("[*] Loading DWPose + ESRGAN ...", flush=True)
    PIPE["dwpose"] = DWPose()
    PIPE["esrgan"] = {
        "ultrasharp": RealESRGANUpscaler(f"{BASE_DIR}/upscale/4x-UltraSharp.pth", "cuda:1"),
        "esrgan": RealESRGANUpscaler(f"{BASE_DIR}/upscale/RealESRGAN_x4plus.pth", "cuda:1"),
    }
    print(f"[*] All loaded in {time.time()-t0:.0f}s", flush=True)


# ---------------- 工具 ----------------
def _gen(pipe, prompt, neg, steps, seed, w, h, gs, **kw):
    g = torch.Generator("cpu").manual_seed(seed)
    with GEN_LOCK:
        img = pipe(prompt=prompt, negative_prompt=neg, num_inference_steps=steps,
                   width=w, height=h, guidance_scale=gs, generator=g, **kw).images[0]
        torch.cuda.empty_cache()  # 释放推理峰值缓存，防碎片 OOM
    return img


def _save(img, tag, seed):
    os.makedirs(OUT_DIR, exist_ok=True)
    p = f"{OUT_DIR}/{tag}_{int(time.time())}_{seed}.png"
    img.save(p)
    return p


def _imread(path):
    return cv2.imread(path)


# ---------------- 接口 ----------------
@app.get("/health")
def health():
    return {"status": "ok", "loaded": sorted(PIPE.keys())}


@app.post("/generate")
def generate(req: GenReq):
    t0 = time.time()
    img = _gen(PIPE["base"], req.prompt, req.negative_prompt, req.steps,
               req.seed, req.width, req.height, req.guidance_scale)
    p = _save(img, "base", req.seed)
    return {"image": p, "seconds": round(time.time() - t0, 1)}


@app.post("/pose")
def pose(image: str = Body(..., embed=True)):
    img = _imread(image)
    dets = PIPE["dwpose"].detect_full(img)
    out = []
    for d in dets:
        kps = d["kps"]
        out.append({
            "bbox": [round(float(v), 1) for v in d["bbox"][:4]],
            "conf": round(float(d["bbox"][4]), 3),
            "body": int((kps[:17, 2] > 0.3).sum()),
            "hands": PIPE["dwpose"].hand_boxes(kps, img.shape[:2]),
        })
    canvas = img.copy()
    for d in dets:
        sk = PIPE["dwpose"].draw_skeleton(np.zeros_like(img), d["kps"])
        canvas = cv2.addWeighted(canvas, 0.6, sk, 0.9, 0)
    p = f"{OUT_DIR}/pose_{int(time.time())}.png"
    cv2.imwrite(p, canvas)
    return {"persons": len(out), "detail": out, "pose_image": p}


@app.post("/cn_generate")
def cn_generate(req: CNReq):
    t0 = time.time()
    cn_name = req.condition if req.condition in ("openpose", "depth", "canny") else "openpose"
    # 条件图来源
    if req.image:
        cond = _imread(req.image)
    elif req.source_image:
        src = _imread(req.source_image)
        if cn_name == "openpose":
            dets = PIPE["dwpose"].detect_full(src)
            canvas = np.zeros_like(src)
            for d in dets:
                canvas = PIPE["dwpose"].draw_skeleton(canvas, d["kps"])
            cond = canvas
        elif cn_name == "canny":
            cond = cv2.Canny(src, 80, 200)
            cond = cv2.cvtColor(cond, cv2.COLOR_GRAY2BGR)
        else:
            return {"error": "depth 需提供 image（MiDaS 未内置）"}
    else:
        return {"error": "需提供 image 或 source_image"}
    cond_rgb = cv2.cvtColor(cond, cv2.COLOR_BGR2RGB)
    cond_img = __import__("PIL").Image.fromarray(cond_rgb)
    # 懒加载非 openpose ControlNet（重建 cn_pipe，释放旧显存）
    if cn_name not in PIPE["cn_names"]:
        load_cn(cn_name)
    img = _gen(PIPE["cn_pipe"], req.prompt, req.negative_prompt, req.steps,
               req.seed, req.width, req.height, req.guidance_scale,
               image=[cond_img],  # MultiControlNetModel 要求 list
               controlnet_conditioning_scale=[req.cn_strength])
    p = _save(img, f"cn_{cn_name}", req.seed)
    cond_p = f"{OUT_DIR}/cn_{cn_name}_cond_{int(t0)}.png"
    cv2.imwrite(cond_p, cond)
    return {"image": p, "condition": cond_p, "seconds": round(time.time() - t0, 1)}


@app.post("/inpaint")
def inpaint(req: InpaintReq):
    t0 = time.time()
    img = _imread(req.image)
    h, w = img.shape[:2]
    if req.hand_box:
        x1, y1, x2, y2 = [int(v) for v in req.hand_box.split(",")]
    else:
        dets = PIPE["dwpose"].detect_full(img)
        boxes = []
        for d in dets:
            boxes += PIPE["dwpose"].hand_boxes(d["kps"], (h, w))
        if not boxes:
            return {"error": "未检测到手部区域"}
        x1, y1, x2, y2 = max(0, min(b[0] for b in boxes)), max(0, min(b[1] for b in boxes)), \
                         min(w, max(b[2] for b in boxes)), min(h, max(b[3] for b in boxes))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)  # 边缘羽化
    pil_img = __import__("PIL").Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    pil_mask = __import__("PIL").Image.fromarray(mask)
    g = torch.Generator("cpu").manual_seed(req.seed)
    with GEN_LOCK:
        out = PIPE["inpaint"](prompt=req.prompt, negative_prompt=req.negative_prompt,
                              image=pil_img, mask_image=pil_mask,
                              num_inference_steps=req.steps, guidance_scale=7.5,
                              strength=req.denoise, generator=g).images[0]
    p = _save(out, "inpaint", req.seed)
    mask_p = f"{OUT_DIR}/inpaint_mask_{int(t0)}.png"
    cv2.imwrite(mask_p, mask)
    return {"image": p, "mask": mask_p, "hand_box": [x1, y1, x2, y2],
            "seconds": round(time.time() - t0, 1)}


@app.post("/upscale")
def upscale(req: UpscaleReq):
    t0 = time.time()
    img = _imread(req.image)
    up = PIPE["esrgan"].get(req.model, PIPE["esrgan"]["ultrasharp"])
    out = up.upscale(img)
    p = f"{OUT_DIR}/upscaled_{int(t0)}.png"
    cv2.imwrite(p, out)
    return {"image": p, "size": list(out.shape[:2]),
            "seconds": round(time.time() - t0, 1)}


@app.post("/pipeline")
def pipeline(req: PipeReq):
    """端到端：txt2img → DWPose 检测 → ControlNet 重生成 → inpaint 修手 → 超分"""
    t0 = time.time()
    steps = []
    # 1. 首轮生成（无 ControlNet，用提示词方案）
    img = _gen(PIPE["base"], req.prompt, req.negative_prompt, req.steps,
               req.seed, req.width, req.height, req.guidance_scale)
    p1 = _save(img, "pipe_1_txt2img", req.seed)
    steps.append({"step": 1, "name": "txt2img", "image": p1, "t": round(time.time() - t0, 1)})

    # 2. ControlNet OpenPose 重生成（姿态约束）
    if req.use_cn:
        src = _imread(p1)
        dets = PIPE["dwpose"].detect_full(src)
        if dets:
            canvas = np.zeros_like(src)
            for d in dets:
                canvas = PIPE["dwpose"].draw_skeleton(canvas, d["kps"])
            cond_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            cond_img = __import__("PIL").Image.fromarray(cond_rgb)
            img = _gen(PIPE["cn_pipe"], req.prompt, req.negative_prompt, req.steps,
                       req.seed, req.width, req.height, req.guidance_scale,
                       image=[cond_img],  # MultiControlNetModel 要求 list
                       controlnet_conditioning_scale=[req.cn_strength])
            p2 = _save(img, "pipe_2_controlnet", req.seed)
            cond_p = f"{OUT_DIR}/pipe_2_cond_{int(t0)}.png"
            cv2.imwrite(cond_p, canvas)
            steps.append({"step": 2, "name": "controlnet_openpose", "image": p2,
                          "condition": cond_p, "t": round(time.time() - t0, 1)})

    # 3. 检测手部区域 + inpaint 局部重绘
    if req.use_inpaint:
        cur = _imread(steps[-1]["image"])
        h, w = cur.shape[:2]
        dets = PIPE["dwpose"].detect_full(cur)
        boxes = []
        for d in dets:
            boxes += PIPE["dwpose"].hand_boxes(d["kps"], (h, w))
        if boxes:
            x1 = max(0, min(b[0] for b in boxes)); y1 = max(0, min(b[1] for b in boxes))
            x2 = min(w, max(b[2] for b in boxes)); y2 = min(h, max(b[3] for b in boxes))
            mask = np.zeros((h, w), np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            pil_img = __import__("PIL").Image.fromarray(cv2.cvtColor(cur, cv2.COLOR_BGR2RGB))
            pil_mask = __import__("PIL").Image.fromarray(mask)
            g = torch.Generator("cpu").manual_seed(req.seed)
            with GEN_LOCK:
                img = PIPE["inpaint"](prompt="perfect detailed hand, five fingers, correct anatomy",
                                      negative_prompt=req.negative_prompt, image=pil_img,
                                      mask_image=pil_mask, num_inference_steps=req.steps,
                                      guidance_scale=7.5, strength=req.inpaint_denoise,
                                      generator=g).images[0]
            p3 = _save(img, "pipe_3_inpaint", req.seed)
            steps.append({"step": 3, "name": "inpaint_hand", "image": p3,
                          "hand_box": [x1, y1, x2, y2], "t": round(time.time() - t0, 1)})

    # 4. 全局超分（最后一步，防畸形放大）
    if req.use_upscale:
        cur = _imread(steps[-1]["image"])
        out = PIPE["esrgan"]["ultrasharp"].upscale(cur)
        p4 = f"{OUT_DIR}/pipe_4_upscaled_{int(t0)}.png"
        cv2.imwrite(p4, out)
        steps.append({"step": 4, "name": "upscale_x4", "image": p4,
                      "size": list(out.shape[:2]), "t": round(time.time() - t0, 1)})

    return {"steps": steps, "total_seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10335)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
