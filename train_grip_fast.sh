#!/bin/bash
#SBATCH --job-name=train_region_rq
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
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
CKPT_DIR="checkpoints/$CONFIG/region_v1"

# Wait for dataset + norm_stats (max 4 hours, then give up instead of holding GPUs).
echo ">>> GPUs allocated ($(date)); waiting for dataset + norm_stats..."
WAITED=0
until [ -f "$DATA" ] && [ -f "$NORM" ]; do
    sleep 60; WAITED=$((WAITED+1))
    [ $WAITED -ge 240 ] && { echo "!!! Waited over 4 hours, giving up"; exit 1; }
    [ $((WAITED % 5)) -eq 0 ] && echo "  ... waited ${WAITED} min (data:$([ -f "$DATA" ]&&echo ok||echo -) norm:$([ -f "$NORM" ]&&echo ok||echo -))"
done
echo ">>> Ready ($(date)), starting training"

# Preemption-safe: if a previous (preempted) run left checkpoints, resume from them;
# otherwise start fresh. Combined with --save-interval=5000, a preemption costs at
# most ~5000 steps of progress and the job auto-requeues and continues.
if [ -d "$CKPT_DIR" ] && [ -n "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
    MODE=--resume
    echo ">>> Found existing checkpoints in $CKPT_DIR -> resuming"
else
    MODE=--overwrite
    echo ">>> No existing checkpoints -> fresh start"
fi

set -e
uv run --no-sync scripts/train.py $CONFIG \
  --exp-name=region_v1 $MODE \
  --save-interval=5000 \
  --keep-period=10000
