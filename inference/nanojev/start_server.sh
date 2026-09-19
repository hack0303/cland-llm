#!/bin/bash
# 启动 NanoJev 常驻决策服务（本机 venv，复用 base 的 torch cu118 / P40）
#
# 用法: ./start_server.sh [variant] [port] [precision]
#   variant   默认 local_atomic_seed17（50×50 maze showcase；snake 用 games_gold_seed17）
#   port      默认 10338
#   precision 默认 fp32（P40 无原生 BF16；bf16 为模拟路径，仅用于对照）
#
# 日志: inference/nanojev/logs/server-<variant>-<port>.log
set -e
cd "$(dirname "$0")"

VARIANT="${1:-local_atomic_seed17}"
PORT="${2:-10338}"
PRECISION="${3:-fp32}"
GPU="${NANOJEV_GPU:-0}"
CKPT="/mnt/data/ai_workspace/models/nanojev/NanoJev/variants/${VARIANT}"

if [ ! -f "$CKPT/best.safetensors" ]; then
  echo "缺少 checkpoint: $CKPT" >&2
  echo "先下载: HF_ENDPOINT=https://hf-mirror.com ./venv/bin/python download_variant.py ${VARIANT}" >&2
  exit 1
fi

if ss -tln 2>/dev/null | grep -q ":${PORT} "; then
  echo "端口 ${PORT} 已在监听，服务可能已运行"
  exit 0
fi

mkdir -p logs
LOG="logs/server-${VARIANT}-${PORT}.log"
cd NanoJev
nohup env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}" \
  ../venv/bin/python scripts/serve_decisions.py \
  --checkpoint-dir "${CKPT}" \
  --web-root web \
  --host 127.0.0.1 --port "${PORT}" \
  --precision "${PRECISION}" \
  > "../${LOG}" 2>&1 &
echo "NanoJev 服务启动中 pid=$! variant=${VARIANT} port=${PORT} precision=${PRECISION}"
echo "日志: inference/nanojev/${LOG}"
echo "健康检查: curl http://127.0.0.1:${PORT}/api/health"
