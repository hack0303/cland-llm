#!/bin/bash
# 启动手脚修复管线服务 (GPU1 :10335)
cd /mnt/data/ai_workspace/cland-llm/inference/sdxl/hand_pipe
exec env TORCHINDUCTOR_COMPILE_THREADS=1 \
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python3 -u pipe_server.py --port 10335
