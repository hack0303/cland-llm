#!/usr/bin/env python3
"""通过 HF 镜像下载 von-1.0 权重至本机模型盘。"""
import os, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
t0 = time.time()
path = snapshot_download(
    repo_id="wfzyx/von-1.0",
    local_dir="/mnt/data/ai_workspace/models/von/von-1.0",
)
print(f"[done] von-1.0 path={path} elapsed={time.time()-t0:.1f}s", flush=True)
