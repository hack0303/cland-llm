#!/usr/bin/env python3
"""TripoSG 图生 3D 常驻 API 服务 (GPU 1 / 端口 10332)
启动: CUDA_VISIBLE_DEVICES=1 python3 server.py --port 10332
"""
import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import trimesh
from fastapi import FastAPI, File, Form, UploadFile
import uvicorn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, "scripts"))
sys.path.append(SCRIPT_DIR)

from triposg.pipelines.pipeline_triposg import TripoSGPipeline  # noqa: E402
from image_process import prepare_image  # noqa: E402
from briarmbg import BriaRMBG  # noqa: E402

WEIGHTS = os.path.join(SCRIPT_DIR, "pretrained_weights")
OUT_DIR = "/mnt/data/ai_workspace/outputs3d"
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="TripoSG 3D Service")


@app.on_event("startup")
def load_model():
    global pipe, rmbg_net
    t0 = time.time()
    print(f"[*] Loading TripoSG from {WEIGHTS} ...", flush=True)
    rmbg_net = BriaRMBG.from_pretrained(os.path.join(WEIGHTS, "RMBG-1.4")).to("cuda")
    rmbg_net.eval()
    pipe = TripoSGPipeline.from_pretrained(
        os.path.join(WEIGHTS, "TripoSG")
    ).to("cuda", torch.float16)
    print(f"[*] Model loaded in {time.time()-t0:.0f}s, "
          f"VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "TripoSG", "gpu": torch.cuda.get_device_name(0)}


@app.post("/generate")
def generate(
    image: UploadFile = File(...),
    steps: int = Form(50),
    seed: int = Form(42),
    guidance_scale: float = Form(7.0),
    faces: int = Form(-1),
):
    img_bytes = image.file.read()
    tmp_img = f"/tmp/triposg_upload_{int(time.time())}.png"
    with open(tmp_img, "wb") as f:
        f.write(img_bytes)
    img_pil = prepare_image(
        tmp_img, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=rmbg_net
    )
    os.remove(tmp_img)
    t0 = time.time()
    with torch.no_grad():
        outputs = pipe(
            image=img_pil,
            generator=torch.Generator(device=pipe.device).manual_seed(seed),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
        ).samples[0]
        mesh = trimesh.Trimesh(
            outputs[0].astype(np.float32), np.ascontiguousarray(outputs[1])
        )
    if faces > 0 and mesh.faces.shape[0] > faces:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces)
        m = ms.current_mesh()
        mesh = trimesh.Trimesh(vertices=m.vertex_matrix(), faces=m.face_matrix())
    dt = time.time() - t0
    out_path = os.path.join(OUT_DIR, f"triposg_{int(t0)}_{seed}.glb")
    mesh.export(out_path)
    return {
        "model": "TripoSG",
        "glb": out_path,
        "seconds": round(dt, 1),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "seed": seed,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=10332)
    args = ap.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
