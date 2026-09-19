#!/bin/bash
# 启动 Von 常驻决策服务（TypeSafe /v1/systemone 协议对等，drop-in）
#
# 用法: ./start_von.sh [port] [device]
#   port    默认 10339
#   device  默认 cuda（VON_DEVICE 透传；P40 上 bf16 由 torch 模拟路径支撑）
#
# 权重: /mnt/data/ai_workspace/models/von/von-1.0
#       通过 von/checkpoints/von-modernbert-rlcd 符号链接被 von 本地加载（不联网）
# 日志: inference/nanojev/logs/von-server-<port>.log
set -e
cd "$(dirname "$0")"

PORT="${1:-10339}"
DEVICE="${2:-cuda}"
MODEL_DIR="/mnt/data/ai_workspace/models/von/von-1.0"
GPU="${VON_GPU:-0}"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
  echo "缺少 von-1.0 权重: $MODEL_DIR" >&2
  echo "先下载: HF_ENDPOINT=https://hf-mirror.com ./venv/bin/python download_von.py" >&2
  exit 1
fi

mkdir -p logs
mkdir -p von/checkpoints
if [ ! -e von/checkpoints/von-modernbert-rlcd ]; then
  ln -s "$MODEL_DIR" von/checkpoints/von-modernbert-rlcd
fi

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已在监听，服务可能已运行"
  exit 0
fi

LOG="logs/von-server-${PORT}.log"
cd von
nohup env CUDA_VISIBLE_DEVICES="${GPU}" VON_DEVICE="${DEVICE}" \
  ../venv/bin/von serve --host 127.0.0.1 --port "${PORT}" \
  > "../${LOG}" 2>&1 &
echo "Von 服务启动中 pid=$! port=${PORT} device=${DEVICE}"
echo "日志: inference/nanojev/${LOG}"
echo "健康检查: curl http://127.0.0.1:${PORT}/health"
