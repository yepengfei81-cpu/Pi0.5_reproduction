#!/bin/bash
#SBATCH --job-name=single_pair
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200|h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=slurm-%j.out
#
# 一个作业里先后训两个单任务/单爪(50 条平行爪)对照模型: 擦黑板 -> 挂杯, 各 20k 步(2 卡 H200 约 5h/个)。
# 被抢占重启后: 已有终档(19999)的跳过, 没训完的从最新档续。用法: sbatch train_single_pair.sh

export LD_LIBRARY_PATH=$HOME/.conda/envs/ffmpeg/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=/n/home08/yxma/ypf/pi_data
cd ~/ypf/Pi0.5_reproduction

RUNS=(
  "pi05_single_parallel_wipe single_parallel_wipe wipe_single_v1"
  "pi05_single_parallel_mug  single_parallel_mug  mug_single_v1"
)
FINAL_STEP=19999          # = num_train_steps - 1 (配置 20000)
NGPU=${SLURM_GPUS_ON_NODE:-1}
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
REMAT=()
if [ "$NGPU" -le 1 ] || ! echo "$GPU_NAME" | grep -qi "H200"; then
    REMAT=(--model.llm-remat-policy nothing_saveable)
fi
echo ">>> job $SLURM_JOB_ID: ${NGPU} 卡 / ${GPU_NAME} ($(date)) remat=${REMAT[*]:-config}"

for run in "${RUNS[@]}"; do
    set -- $run; CONFIG=$1; REPO=$2; EXP=$3
    # ONLY=<exp> 只跑其中一个(拆成两个独立作业, 各要 1 卡 ~11h, 更容易回填进空档):
    #   sbatch --export=ALL,ONLY=wipe_single_v1 --gres=gpu:1 --time=11:00:00 -c 8 --mem=64G train_single_pair.sh
    #   sbatch --export=ALL,ONLY=mug_single_v1  --gres=gpu:1 --time=11:00:00 -c 8 --mem=64G train_single_pair.sh
    if [ -n "${ONLY:-}" ] && [ "$ONLY" != "$EXP" ]; then
        echo ">>> [$EXP] ONLY=$ONLY, 跳过"; continue
    fi
    DATA="$HF_LEROBOT_HOME/$REPO/meta/info.json"
    NORM="assets/$CONFIG/$REPO/norm_stats.json"
    CKPT_DIR="checkpoints/$CONFIG/$EXP"
    if [ -d "$CKPT_DIR/$FINAL_STEP" ]; then
        echo ">>> [$EXP] 终档已存在, 跳过"; continue
    fi
    echo ">>> [$EXP] 等待数据集 + norm_stats..."
    WAITED=0
    until [ -f "$DATA" ] && [ -f "$NORM" ]; do
        sleep 60; WAITED=$((WAITED+1))
        [ $WAITED -ge 240 ] && { echo "!!! [$EXP] 等了 4 小时没等到, 放弃"; exit 1; }
        [ $((WAITED % 5)) -eq 0 ] && echo "  ... ${WAITED} min (data:$([ -f "$DATA" ]&&echo ok||echo -) norm:$([ -f "$NORM" ]&&echo ok||echo -))"
    done
    if ls "$CKPT_DIR" 2>/dev/null | grep -qE '^[0-9]+$'; then
        MODE=--resume; echo ">>> [$EXP] 发现已有档位 -> resuming"
    else
        MODE=--overwrite; echo ">>> [$EXP] fresh start"
    fi
    echo ">>> [$EXP] 开始训练 ($(date))"
    uv run --no-sync scripts/train.py $CONFIG \
      --exp-name=$EXP $MODE \
      --save-interval=2000 \
      --keep-period=None \
      "${REMAT[@]}" || { echo "!!! [$EXP] 训练退出非零 ($(date))"; exit 1; }
    echo ">>> [$EXP] 完成 ($(date))"
done
echo ">>> 两个都完成 ($(date))"
