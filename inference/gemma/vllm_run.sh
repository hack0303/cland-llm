#!/bin/bash
vllm serve \
/mnt/data/ai_workspace/models/gemma-4-26B-A4B-it-AWQ-8bit \
--tensor-parallel-size 2 \
--max-model-len 131072 \
--gpu-memory-utilization 0.95 \
--host 0.0.0.0 \
--port 10303
