#!/bin/bash
#SBATCH --job-name=train_dualarm
#SBATCH --partition=gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out

export LD_LIBRARY_PATH=$HOME/.conda/envs/ffmpeg/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=/n/home08/yxma/ypf/pi_data
cd ~/ypf/Pi0.5_reproduction

CONFIG=pi05_cotrain_dualarm
DATA="$HF_LEROBOT_HOME/cotrain_dualarm/meta/info.json"
NORM="assets/$CONFIG/cotrain_dualarm/norm_stats.json"

echo ">>> 已分到卡 ($(date)); 等待数据 + norm_stats 就绪..."
WAITED=0
until [ -f "$DATA" ] && [ -f "$NORM" ]; do
    sleep 60; WAITED=$((WAITED+1))
    [ $WAITED -ge 240 ] && { echo "!!! 等待超4小时, 放弃"; exit 1; }
    [ $((WAITED % 5)) -eq 0 ] && echo "  ...已等 ${WAITED} 分钟 (data:$([ -f "$DATA" ]&&echo ok||echo -) norm:$([ -f "$NORM" ]&&echo ok||echo -))"
done
echo ">>> 就绪 ($(date)), 开始训练"

set -e
uv run --no-sync scripts/train.py $CONFIG \
  --exp-name=dualarm_v1 \
  --overwrite \
  --save-interval=999999 \
  --keep-period=999999