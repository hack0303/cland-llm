#!/bin/bash
# NanoJev / Von 一键复现部署：克隆源码 → venv → von 安装 →（可选）下载全部权重。
#
# 用法: ./setup.sh [--with-weights]
#   --with-weights  同时下载 maze/snake/von/Qwen 基线/dataset（约 7 GB，HF 镜像，可断点续传）
set -e
cd "$(dirname "$0")"
HERE="$PWD"

# 1) 上游源码（实测：GitHub https 克隆不通，走 SSH）
[ -d NanoJev/.git ] || git clone --depth 1 git@github.com:TianyuCodings/NanoJev.git
[ -d von/.git ]     || git clone --depth 1 git@github.com:wfzyx/von.git

# 1b) P40 补丁：VON_DTYPE 显式覆盖 dtype（默认 fp32；不改补丁则 CUDA bf16 模拟会精度回退）
if git -C von apply --check patches/von-p40-fp32.patch 2>/dev/null; then
  git -C von apply patches/von-p40-fp32.patch
  echo "已应用 patches/von-p40-fp32.patch"
fi

# 2) venv（复用 base conda 的 torch 2.7.1+cu118；P40 可用档位）
[ -d venv ] || /home/alice/miniconda3/bin/python -m venv --system-site-packages venv

# 3) von 可编辑安装
venv/bin/pip install -e ./von -q

# 4) 可选：权重下载（HF 直连不通 → 镜像；断连后重跑同命令即可续传）
if [ "$1" = "--with-weights" ]; then
  export HF_ENDPOINT=https://hf-mirror.com
  venv/bin/python download_variant.py local_atomic_seed17
  venv/bin/python download_variant.py games_gold_seed17
  venv/bin/python download_von.py
  venv/bin/python download_qwen.py
  venv/bin/python download_dataset.py
fi

echo "环境就绪。启动服务: ./start_server.sh [variant] [port] [precision] / ./start_von.sh [port]"
echo "基准复现: BENCH_GPU=1 ./run_bench_maze.sh both fp32 / BENCH_GPU=1 ./run_bench_snake.sh both fp32"
