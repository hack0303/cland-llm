#!/usr/bin/env python3
"""通过 HF 镜像下载 NanoJev-Data 的 games_v4/arcade 小包（含 12×12 showcase cohort + 官方回放）。"""
import os, time, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
patterns = sys.argv[1:] or ["games_v4/arcade/*", "games_v4/README.md", "games_v4/manifest.json", "games_v4/verify_dataset.py"]
t0 = time.time()
path = snapshot_download(repo_id="C-Tianyu/NanoJev-Data", repo_type="dataset",
                         local_dir="/mnt/data/ai_workspace/models/nanojev/NanoJev-Data",
                         allow_patterns=patterns)
print(f"[done] dataset path={path} elapsed={time.time()-t0:.1f}s", flush=True)
