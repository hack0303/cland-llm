#!/usr/bin/env python3
"""通过 HF 镜像下载 NanoJev variant 权重（小样：单 variant）。"""
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download

variant = sys.argv[1] if len(sys.argv) > 1 else "local_atomic_seed17"
repo = "C-Tianyu/NanoJev"
dest = "/mnt/data/ai_workspace/models/nanojev/NanoJev"
t0 = time.time()
path = snapshot_download(
    repo_id=repo,
    local_dir=dest,
    allow_patterns=[f"variants/{variant}/*", "GAMES_MODEL_MANIFEST.json", "MODEL_MANIFEST.json"],
)
print(f"[done] variant={variant} path={path} elapsed={time.time()-t0:.1f}s", flush=True)
