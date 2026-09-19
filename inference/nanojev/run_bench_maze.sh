#!/bin/bash
# 50×50 maze showcase 复现（NanoJev local_atomic_seed17 + 未调 Qwen3-0.6B 对照）
#
# 用法: ./run_bench_maze.sh [nanojev|qwen|both] [precision]
#   precision 默认 fp32（P40 无原生 BF16；bf16 为模拟路径）
# 输出: runs/maze_*.json
set -e
cd "$(dirname "$0")"
# P40/torch 2.7.1 兼容 shim（torch._native.triton_utils no-op，见 sitecustomize.py）
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
WHAT="${1:-both}"
PRECISION="${2:-fp32}"
GPU="${BENCH_GPU:-0}"
mkdir -p runs

EPISODES="NanoJev/results/side_by_side_maze_episode.jsonl"   # 50×50 showcase: maze:ood:50:24310922
CKPT="/mnt/data/ai_workspace/models/nanojev/NanoJev/variants/local_atomic_seed17"

run_nanojev() {
  CUDA_VISIBLE_DEVICES="$GPU" venv/bin/python NanoJev/scripts/evaluate_model_edges_maze.py \
    --episodes "$EPISODES" \
    --engine checkpoint \
    --checkpoint "$CKPT" \
    --max-steps 0 --batch-states 2 --batch-questions 0 --max-length 2048 \
    --precision "$PRECISION" \
    --output "runs/maze_nanojev_local_atomic_${PRECISION}.json"
}

run_qwen() {
  CUDA_VISIBLE_DEVICES="$GPU" venv/bin/python NanoJev/scripts/evaluate_native_qwen_maze.py \
    --episodes "$EPISODES" \
    --splits ood --window-size 5 --max-steps 0 --batch-states 2 --max-length 2048 \
    --precision "$PRECISION" \
    --output "runs/maze_qwen_native_${PRECISION}.json"
}

case "$WHAT" in
  nanojev) run_nanojev ;;
  qwen) run_qwen ;;
  both) run_nanojev; run_qwen ;;
  *) echo "用法: $0 [nanojev|qwen|both] [fp32|bf16]" >&2; exit 1 ;;
esac
