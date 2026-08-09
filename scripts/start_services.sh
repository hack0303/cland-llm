#!/bin/bash
# C-Land 推理服务一键启动脚本（机器重启后恢复所有常驻服务）
# 用法: bash scripts/start_services.sh [--wait]
set -e
cd /mnt/data/ai_workspace/cland-llm
source /home/alice/miniconda3/etc/profile.d/conda.sh

LOG=/tmp/services_restart.log
echo "=== $(date) restarting services ===" | tee $LOG

start() {  # $1=env  $2=dir  $3=port  $4=logfile  $5=gpu
  local env=$1 dir=$2 port=$3 lf=$4 gpu=$5
  if curl -s --max-time 2 http://127.0.0.1:$port/health > /dev/null 2>&1; then
    echo "[$port] already up" | tee -a $LOG
    return
  fi
  conda activate $env
  export PATH="$HOME/bin:$PATH"
  cd /mnt/data/ai_workspace/cland-llm/$dir
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 server.py --port $port > $lf 2>&1 &
  echo "[$port] started (pid $!, env=$env, gpu=$gpu)" | tee -a $LOG
  conda deactivate
}

start base  inference/sdxl      10331 /tmp/sdxl_server.log      0
start triposg_env inference/triposg 10332 /tmp/triposg_server.log 1
start audio_env inference/tts   10333 /tmp/tts_server.log      1
start audio_env inference/asr   10334 /tmp/asr_server.log      1

# SFX（AudioGen）默认不启动：机器仅 15GB RAM，5 服务同时常驻会 OOM。
# 需要时: bash scripts/start_services.sh --with-sfx
if [ "$1" == "--with-sfx" ]; then
  start audio_env inference/sfx   10336 /tmp/sfx_server.log      1
fi

if [ "$1" == "--wait" ]; then
  echo "--- waiting for health ---" | tee -a $LOG
  for i in $(seq 1 60); do
    for p in 10331 10332 10333 10334 10336; do
      s=$(curl -s --max-time 2 http://127.0.0.1:$p/health 2>/dev/null)
      echo "$(date +%H:%M:%S) :$p $s" | tee -a $LOG
    done
    sleep 20
  done
fi
echo "=== done ===" | tee -a $LOG
