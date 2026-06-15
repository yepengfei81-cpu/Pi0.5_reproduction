# AirBot Play × openpi (π0.5) 复现与部署指南

本仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi) 复现 **π0.5 (pi05)**
视觉-语言-动作模型,并适配 **AIRBOT Play** 机械臂。包含两条端到端管线:

- **管线 A — 关节空间、单任务**:遥操作采集 → 服务器训练 → 本地实时推理(`local_inference.py`,配置 `pi05_airbot_play`)。
- **管线 B — 任务空间(EEF)、UMI + 遥操作联合训练**:把手持 **UMI** 数据和 AIRBOT 遥操作数据转换到**同一套共享任务空间表示**,联合训练后部署(`local_inference_eef.py`,配置 `pi05_cotrain_eef`)。这是当前重点:用手持 UMI 夹爪**廉价**采集新技能(如 *擦黑板*),注入到同时也用遥操作任务(如 *积木放碗*)训练的机器人里。

> English version: [README.md](README.md)。上游 openpi 原始文档:[README_openpi.md](README_openpi.md)。

---

## 0. 总览

### 两类机器、两套环境
| 机器 | 环境工具 | 用途 |
|---|---|---|
| **本地工作站**(机械臂 + 相机 + GPU) | **conda**(一个环境装全) | 采集数据、本地推理 |
| **训练服务器**(大显存 GPU,无机械臂) | **uv** | 训练、算 norm_stats |

本地推理脚本要在**同一进程**里同时加载 openpi 模型和机械臂 SDK,conda 一锅端最省事;
服务器只跑训练,用 openpi 官方推荐的 uv。

### 硬件(本套配置,换硬件需相应修改)
- **机械臂 ×2**:Lead(Replay 无动力主臂,gRPC `50051`)+ Follow(Play 有动力从臂,`50050`)。
- **相机 ×2**:RealSense D405 —— head/环境相机 SN `230422271972`,wrist/手眼相机 SN `230422271433`。
- **GPU**:推理 ≥8GB;LoRA 微调 ≥22.5GB。
- **系统**:Ubuntu 20.04+。

相机序列号在 [airbot/play_config.json](airbot/play_config.json) 和 [airbot/collect_data.py](airbot/collect_data.py)。

---

## 1. 机械臂 & 相机环境(系统级,仅真机端需要)

AIRBOT 专有软件**不在本仓库**。Ubuntu + 纯 Python 只需:

| 文件 | 用途 |
|---|---|
| `airbot-configure_5.1.6-1_all.deb` | 驱动(CAN / udev) |
| `airbot_py-5.1.6-py3-none-any.whl` | Python SDK |

```bash
sudo apt-get install ./airbot-configure_5.1.6-1_all.deb   # 驱动
pip install ./airbot_py-5.1.6-py3-none-any.whl            # SDK(装进 §2 的 conda 环境)
pip install pyrealsense2                                  # 相机
```

机械臂封装 [airbot/play_sdk.py](airbot/play_sdk.py) 已在仓库内。

---

## 2. 本地环境(conda)—— 采集 + 推理

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
conda create -n openpi_airbot python=3.11 -y
conda activate openpi_airbot

pip install -e packages/openpi-client
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pip install -e .
pip install ./airbot_py-5.1.6-py3-none-any.whl pyrealsense2
```

系统级 NVIDIA 驱动 / CUDA 需自行装好(`nvidia-smi` 正常)。

---

## 3. 服务器环境(uv)—— 训练

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules <repo-url> && cd <repo>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
后续训练命令统一用 `uv run python ...`。

> 服务器需要 `ffmpeg` 解码视频(`apt-get install ffmpeg`);本仓库的 LeRobot 数据集相机以 H.264 视频存储。

---

## 4. 管线 A —— 关节空间、单任务

state/action = 6 关节 + 夹爪(7D),绝对关节角。

```bash
# 采集(本地)
python airbot/collect_data.py --task "pick up the block and place it in the bowl"

# 训练(服务器)
export HF_LEROBOT_HOME=/path/to/bigdisk/data
uv run scripts/compute_norm_stats.py --config-name pi05_airbot_play
uv run scripts/train.py pi05_airbot_play --exp-name=run1 --num-train-steps=30000 --batch-size=8

# 推理(本地)—— 先 dry-run
python local_inference.py --checkpoint ./checkpoints/pi05_airbot_play/run1/30000 \
  --task "pick up the block and place it in the bowl" --dry-run
```
安全:`q`/`Esc`(或 `--no-display` 下终端 `q`+回车)急停回零;`--min-z` 限制末端最低高度防压台面。

---

## 5. 管线 B —— 任务空间(EEF)、UMI + 遥操作联合训练

目标:用**手持 UMI 夹爪**廉价采集新技能并注入真机,和 AIRBOT 遥操作数据联合训练,
方法是把两类来源都映射到**同一套共享任务空间表示**。

### 5.1 共享表示
每条样本(UMI 与遥操作)都转成同一 schema:

- **state / action = 10D = `pos(3) + rot6d(6) + 夹爪(1)`**,在每条 episode 的相对系 **W′**
  下(原点 = 首帧指尖,z = 重力,首帧 yaw 归零;pitch/roll 保留为重力绝对)。两类来源因此都"无根"、与本体无关。
- **夹爪**按机械臂最大开度归一化(÷0.073,clip [0,1]),0 = 闭合。动作通道用 **lead(指令)夹爪**,
  这样抓取能表达"夹紧力"(而不是只到物体宽度的实际开度)。
- **两个相机槽**:`wrist_image`(始终有效)+ `image`(环境/头部相机)。
- **`env_mask`**:1 = 环境相机有效(遥操作),0 = 屏蔽(UMI 无全局相机)。让单一策略能在**异构传感器**来源上联合训练。
- **相机视场对齐**:UMI 鱼眼去畸变到 ~79°,对齐 D405 的 78.6°。

### 5.2 步骤
```bash
# (a) 采集遥操作 EEF 数据 —— 通过厂家 FK get_end_pose 记录 base 系末端(pos+quat+夹爪)。
#     lead→follow 夹爪量程修正已内置(写死),无需标定。
python airbot/collect_data.py --task "pick up the block and place it in the bowl" \
  --output-dir /home/you/pi_data --repo-id airbot_play_data
python airbot/collect_data.py --task "wipe the blackboard" \
  --output-dir /home/you/pi_data --repo-id airbot_wipe_data

# (b) 把 UMI mcap + (一个或多个)遥操作数据集打包成一个 EEF 数据集。
#     每条遥操作 episode 的 prompt 自动从源 tasks.jsonl 读;
#     datasets-v4 的 "List" 特征类型自动降级以兼容服务器。
python umi/pack_lerobot.py \
  --umi-dir /path/to/umi_mcaps \
  --teleop-dir /home/you/pi_data/airbot_play_data /home/you/pi_data/airbot_wipe_data \
  -o /home/you/pi_data/cotrain_eef --repo-id cotrain_eef --verify
#   (--skip-umi 只打包遥操作, 供消融配置 pi05_teleop_eef)

# (c) 训练(服务器)
uv run scripts/compute_norm_stats.py --config-name pi05_cotrain_eef
uv run scripts/train.py pi05_cotrain_eef --exp-name cotrain_v1

# (d) 部署(本地)。env-mask=1 用头部相机(任务有遥操作数据后推荐);
#     单相机的纯 UMI 任务用 env-mask=0。
python local_inference_eef.py --checkpoint ./checkpoints/pi05_cotrain_eef/cotrain_v1/17000 \
  --task "pick up the block and place it in the bowl" --env-mask 1 \
  --speed-profile fast --speed-scale 0.3
```

### 5.3 离线开环评测(不接机器人)
通过喂入录制好的观测、对比预测动作 vs 真值动作,验证模型是否真学会了某条 episode
(flow-matching 的低 loss **不**保证采样动作好):
```bash
python eval_eef_offline.py --checkpoint <ckpt> --dataset /home/you/pi_data/cotrain_eef \
  --task-filter "block" --plot eval_block.png
#  --force-env-mask 0/1 可探测对环境相机掩码的敏感度
```

> 部署是闭环的:每个 chunk 都用实时 `get_end_pose` 重新推理,可自纠错。注意:被错误执行的
> 夹爪维会反馈进观测状态、进而污染位姿预测——所以夹爪反归一化和"动作用 lead 夹爪"影响的是整条策略,不只夹爪本身。

---

## 6. 数据集 & 权重(HuggingFace)

数据集和微调权重**不进 git**。
```bash
hf auth login
hf upload <user>/<dataset> "$HF_LEROBOT_HOME/<dataset>" . --repo-type dataset
cd checkpoints/<config>/<exp>/<step> && hf upload <user>/<weights> . --exclude "train_state/*"
```
`HF_LEROBOT_HOME` 是 LeRobot 数据集根目录;采集、打包、训练都按它定位数据集,训练前务必先 export。

---

## 7. 关键文件
| 路径 | 说明 |
|---|---|
| [airbot/collect_data.py](airbot/collect_data.py) | 遥操作 + 双相机采集 → LeRobot(关节 7D **和** base 系 EEF 8D);lead→follow 夹爪量程映射。 |
| [airbot/play_sdk.py](airbot/play_sdk.py) | AIRBOT Play 机械臂 + 相机封装。 |
| [airbot/play_config.json](airbot/play_config.json) | 端口、相机序列号、工作空间范围。 |
| [umi/umi_to_lerobot.py](umi/umi_to_lerobot.py) | UMI mcap → 去畸变视频 + W′ 位姿 + 夹爪归一化。 |
| [umi/pack_lerobot.py](umi/pack_lerobot.py) | 阶段4 打包:UMI + 遥操作 → 同一套共享 EEF 数据集。 |
| [umi/replay_check.py](umi/replay_check.py) | 真机回放验证坐标链。 |
| [local_inference.py](local_inference.py) | 关节空间实时推理(管线 A)。 |
| [local_inference_eef.py](local_inference_eef.py) | 任务空间(EEF)实时推理(管线 B)。 |
| [eval_eef_offline.py](eval_eef_offline.py) | 离线开环评测(预测 vs 真值动作)。 |
| `src/openpi/policies/airbot_eef_policy.py` | EEF 数据 ↔ 模型变换(10D state、env_mask)。 |
| `src/openpi/training/config.py` | 配置 `pi05_airbot_play`、`pi05_cotrain_eef`、`pi05_teleop_eef`。 |

---

## 8. 测试后清理
```bash
conda env remove -n openpi_airbot
rm -rf .venv
du -sh ~/.cache/openpi ~/.cache/huggingface   # 跨项目共享缓存, 按需清理
```
