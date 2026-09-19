#!/bin/bash
# Snake showcase 复现（NanoJev games_gold_seed17 + 未调 Qwen3-0.6B 对照）
#
# 用法: ./run_bench_snake.sh [nanojev|qwen|both] [precision]
#   默认 both fp32；输出 runs/snake_*.json
# 备注: 8 局冻结 cohort（含 README showcase snake:showcase:12:61005），greedy 控制器，seed 17。
set -e
cd "$(dirname "$0")"
# P40/torch 2.7.1 兼容 shim（torch._native.triton_utils no-op，见 sitecustomize.py）
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
WHAT="${1:-both}"
PRECISION="${2:-fp32}"
GPU="${BENCH_GPU:-0}"
mkdir -p runs

EPISODES="NanoJev/results/arcade_snake_cohort.jsonl"
CKPT="/mnt/data/ai_workspace/models/nanojev/NanoJev/variants/games_gold_seed17"

run_nanojev() {
  CUDA_VISIBLE_DEVICES="$GPU" venv/bin/python NanoJev/scripts/evaluate_composed_snake.py \
    --episodes "$EPISODES" \
    --engine checkpoint \
    --checkpoint "$CKPT" \
    --controller greedy --max-steps 256 --seed 17 \
    --batch-states 2 --batch-questions 0 --max-length 8192 \
    --precision "$PRECISION" \
    --output "runs/snake_nanojev_games_gold_${PRECISION}.json"
}

run_qwen() {
  CUDA_VISIBLE_DEVICES="$GPU" venv/bin/python NanoJev/scripts/evaluate_composed_snake.py \
    --episodes "$EPISODES" \
    --engine native \
    --controller greedy --max-steps 256 --seed 17 \
    --batch-states 2 --batch-questions 0 --max-length 8192 \
    --precision "$PRECISION" \
    --output "runs/snake_qwen_native_${PRECISION}.json"
}

case "$WHAT" in
  nanojev) run_nanojev ;;
  qwen) run_qwen ;;
  both) run_nanojev; run_qwen ;;
  *) echo "用法: $0 [nanojev|qwen|both] [fp32|bf16]" >&2; exit 1 ;;
esac
