#!/bin/bash
# 启动 reranker 常驻服务（手动启动，无开机自启）
# 用法: ./start_reranker.sh  → 日志 /tmp/reranker-server.log
set -e
cd "$(dirname "$0")"

# GPU1（与 embedding 同卡；GPU0 被 gemma4 26B 占用）
export CUDA_VISIBLE_DEVICES=1

if ss -tln 2>/dev/null | grep -q :10304; then
  echo "reranker 已在运行 (10304)"
  exit 0
fi

nohup /home/alice/miniconda3/envs/dev_bge/bin/python reranker_server.py \
  > /tmp/reranker-server.log 2>&1 &

echo "reranker 启动中 pid=$! (模型加载约 10-20s)"
echo "日志: /tmp/reranker-server.log"
