#!/usr/bin/env python3
"""下载未调 Qwen3-0.6B 基线权重（固定 revision，HF 镜像 → 默认 HF 缓存）。

evaluate_native_qwen_*.py 使用 local_files_only=True + 固定 revision 读取默认缓存，
因此这里必须下载到默认缓存（~/.cache/huggingface）。
"""
import os, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import snapshot_download
t0 = time.time()
path = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    revision="c1899de289a04d12100db370d81485cdf75e47ca",
)
print(f"[done] Qwen3-0.6B path={path} elapsed={time.time()-t0:.1f}s", flush=True)
