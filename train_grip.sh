#!/bin/bash
#SBATCH --job-name=train_region
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

CONFIG=pi05_cotrain_dualarm_region
DATA="$HF_LEROBOT_HOME/cotrain_dualarm2/meta/info.json"
NORM="assets/$CONFIG/cotrain_dualarm2/norm_stats.json"

# GPUs are allocated at this point; wait until dataset + norm_stats are uploaded
# (max 4 hours, then give up instead of holding the GPUs).
echo ">>> GPUs allocated ($(date)); waiting for dataset + norm_stats..."
WAITED=0
until [ -f "$DATA" ] && [ -f "$NORM" ]; do
    sleep 60; WAITED=$((WAITED+1))
    [ $WAITED -ge 240 ] && { echo "!!! Waited over 4 hours, giving up"; exit 1; }
    [ $((WAITED % 5)) -eq 0 ] && echo "  ... waited ${WAITED} min (data:$([ -f "$DATA" ]&&echo ok||echo -) norm:$([ -f "$NORM" ]&&echo ok||echo -))"
done
echo ">>> Ready ($(date)), starting training"

set -e
uv run --no-sync scripts/train.py $CONFIG \
  --exp-name=region_v1 \
  --overwrite \
  --save-interval=999999 \
  --keep-period=999999