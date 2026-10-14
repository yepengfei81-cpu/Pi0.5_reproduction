#!/bin/bash
#SBATCH --job-name=ultra_nodrop
#SBATCH --partition=gpu_requeue
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=slurm-%j.out
#
# "见缝插针"版: gpu_requeue 分区里 H200 或 H100, 1 张或 2 张都接。
# (2026-10-15 实测: gpu_requeue 特性名 h200/h100/a100/rtx6000pro, 时限上限 3 天; gpu 分区没有这些节点,
#  --partition=gpu_requeue,gpu 多分区提交会整体 BadConstraints, 所以只投 gpu_requeue。)
# 想只要 H200: sbatch --constraint=h200 ...; 1 卡建议 --time=36:00:00(剩余步数多时), 2 卡 16h 够。
# 用法(同一脚本投两次, 两个变体共用一个 checkpoint 目录, 靠锁文件协调):
#   sbatch --gres=gpu:2 train_fast_ultra.sh
#   sbatch --gres=gpu:1 train_fast_ultra.sh
# 规则:
#   - 先拿到卡的先训;
#   - 卡多的作业启动时发现卡少的在跑 -> 取消它, 从最新档续训(最多丢 2000 步);
#   - 卡少的作业启动时发现卡多的在跑 -> 让位退出, 并给自己 1 小时后再投一次(卡多的若被抢占, 它来补位);
#   - 终档 39999 存在 -> 任何变体都直接退出, 不再续投。
#   - remat: 只有 "≥2 卡且是 H200(141 GB)" 才用配置里的 dots_saveable(快 20-35%);
#     1 卡(batch 48 全压一张卡)或 H100(80 GB) 一律切回 nothing_saveable 防 OOM。

export LD_LIBRARY_PATH=$HOME/.conda/envs/ffmpeg/lib:$LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_LEROBOT_HOME=/n/home08/yxma/ypf/pi_data
cd ~/ypf/Pi0.5_reproduction

CONFIG=pi05_cotrain_dualarm_nodrop
EXP=cotrain_nodrop_v1
DATA="$HF_LEROBOT_HOME/cotrain_dualarm5/meta/info.json"
NORM="assets/$CONFIG/cotrain_dualarm5/norm_stats.json"
CKPT_DIR="checkpoints/$CONFIG/$EXP"
FINAL_STEP=39999                       # = num_train_steps - 1 (配置 40000)
NGPU=${SLURM_GPUS_ON_NODE:-1}
LOCK="$CKPT_DIR/.ultra_lock"
SELF="${SLURM_SUBMIT_DIR:-$PWD}/train_fast_ultra.sh"

echo ">>> job $SLURM_JOB_ID: ${NGPU} 卡, 分区 $SLURM_JOB_PARTITION, 节点 $SLURMD_NODENAME ($(date))"
mkdir -p "$CKPT_DIR"

if [ -d "$CKPT_DIR/$FINAL_STEP" ]; then
    echo ">>> 终档 $FINAL_STEP 已存在, 训练已完成, 退出"; exit 0
fi

# ---- 协调: 另一个变体是否在跑 ----
if [ -f "$LOCK" ]; then
    read -r OJOB ONGPU < "$LOCK"
    if [ "$OJOB" != "$SLURM_JOB_ID" ]; then
        OSTATE=$(squeue -h -j "$OJOB" -o %T 2>/dev/null | head -1)
        if [ "$OSTATE" = "RUNNING" ]; then
            if [ "$NGPU" -gt "${ONGPU:-0}" ]; then
                echo ">>> 作业 $OJOB(${ONGPU}卡) 在跑, 本作业 ${NGPU} 卡更多 -> 取消它并接管(从最新档续)"
                scancel "$OJOB"; sleep 45
            else
                echo ">>> 作业 $OJOB(${ONGPU}卡) 在跑, 本作业(${NGPU}卡)让位; 1 小时后再投一次补位"
                [ -z "$ULTRA_DRYRUN" ] && sbatch --begin=now+1hour --gres=gpu:$NGPU "$SELF" >/dev/null
                exit 0
            fi
        else
            echo ">>> 锁里的作业 $OJOB 状态 '${OSTATE:-不存在}'(未在跑), 本作业接管"
        fi
    fi
fi
echo "$SLURM_JOB_ID $NGPU" > "$LOCK"
[ -n "$ULTRA_DRYRUN" ] && { echo ">>> ULTRA_DRYRUN: 协调完成, 本作业将训练"; exit 0; }

# ---- 等数据集 + norm_stats(最多 4 小时) ----
echo ">>> 等待数据集 + norm_stats..."
WAITED=0
until [ -f "$DATA" ] && [ -f "$NORM" ]; do
    sleep 60; WAITED=$((WAITED+1))
    [ $WAITED -ge 240 ] && { echo "!!! 等了 4 小时没等到, 放弃"; exit 1; }
    [ $((WAITED % 5)) -eq 0 ] && echo "  ... ${WAITED} min (data:$([ -f "$DATA" ]&&echo ok||echo -) norm:$([ -f "$NORM" ]&&echo ok||echo -))"
done

# ---- 续训判断: 只认纯数字步数目录 ----
if ls "$CKPT_DIR" 2>/dev/null | grep -qE '^[0-9]+$'; then
    MODE=--resume; echo ">>> 发现已有档位 -> resuming"
else
    MODE=--overwrite; echo ">>> 无档位 -> fresh start"
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
REMAT=()
if [ "$NGPU" -le 1 ] || ! echo "$GPU_NAME" | grep -qi "H200"; then
    REMAT=(--model.llm-remat-policy nothing_saveable)
    echo ">>> ${NGPU} 卡 / ${GPU_NAME:-未知卡型}: remat 切回 nothing_saveable(防 OOM)"
else
    echo ">>> ${NGPU} 卡 / ${GPU_NAME}: 用配置里的 dots_saveable"
fi

set -e
echo ">>> 开始训练 ($(date))"
uv run --no-sync scripts/train.py $CONFIG \
  --exp-name=$EXP $MODE \
  --save-interval=2000 \
  --keep-period=None \
  "${REMAT[@]}"
