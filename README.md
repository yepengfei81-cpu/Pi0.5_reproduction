# AirBot Play × openpi (π0.5) 复现与部署指南

本仓库基于 [openpi](https://github.com/Physical-Intelligence/openpi) 复现 **π0.5 (pi05)** 视觉-语言-动作模型，并适配 **AIRBOT Play** 机械臂：从遥操作采集真机数据 → 服务器训练 → 本地实时推理，完整跑通。

> openpi 上游的原始英文文档见 [README_openpi.md](README_openpi.md)。

---

## 0. 总览

### 整套流程
```
[本地工作站]  遥操作采集数据 (collect_data.py)
      │  上传数据集到 HuggingFace
      ▼
[训练服务器]  repack → 算 norm_stats → 训练 (train.py)
      │  上传权重到 HuggingFace
      ▼
[本地工作站]  下载权重 → 实时推理 (local_inference.py)
```

### 两类机器、两套环境
| 机器 | 环境工具 | 用途 |
|------|----------|------|
| **本地工作站**（接机械臂+相机+GPU） | **conda**（方案 A，一锅端） | 采集数据、本地推理 |
| **训练服务器**（大显存 GPU，无机械臂） | **uv** | 训练 |

> 为什么本地用 conda、服务器用 uv：本地推理脚本要在**同一进程**里同时加载 openpi 模型和机械臂 SDK，conda 一个环境装全最省事；服务器只跑训练，用 openpi 官方推荐的 uv 最干净。

### 硬件（本套配置，换硬件需相应修改）
- **机械臂 ×2**：Lead（Replay 无动力主臂，gRPC 端口 `50051`）+ Follow（Play 有动力从臂，端口 `50050`）
- **相机 ×2**：RealSense D405
  - head（环境相机）SN `230422271972`
  - wrist（手眼相机）SN `230422271433`
- **GPU**：推理 ≥8GB；LoRA 微调 ≥22.5GB（详见 [README_openpi.md](README_openpi.md)）
- **系统**：Ubuntu 20.04+

> 相机序列号在 [airbot/play_config.json](airbot/play_config.json) 和 [airbot/collect_data.py](airbot/collect_data.py) 里，换相机改这两处。

---

## 1. 机械臂 & 相机环境（系统级，仅真机端需要）

> 这一步配置 AIRBOT Play 的驱动和 SDK，是采集/推理的前提。**只在接了机械臂的本地工作站上做**，训练服务器不需要。

### 1.1 获取厂家软件包
`airbot-configure`（驱动 .deb）和 `airbot_py`（Python SDK .whl）是 **AIRBOT 专有软件**，不在本仓库内（专有软件不便随 MIT 仓库分发）。

Ubuntu + 纯 Python 使用，**只需以下两个文件**：

| 文件 | 用途 |
|------|------|
| `airbot-configure_5.1.6-1_all.deb` | 驱动（配置 CAN/udev），架构无关 |
| `airbot_py-5.1.6-py3-none-any.whl` | Python SDK |

> `airbot_cpp-*.deb`（C++ SDK）本项目用不到，无需获取。

获取方式：
- **官方途径**：联系 AIRBOT 技术支持获取（不同固件版本配套的驱动不同，务必拿对应版本）；
- **本实验室成员**：从云盘下载 → **📦 下载链接（待填）：`<在此粘贴 Google Drive 链接>`**

### 1.2 安装驱动（配置 CAN 接口 / udev 规则）
```bash
sudo apt-get install ./airbot-configure_5.1.6-1_all.deb
dpkg -l | grep airbot          # 看到版本号即成功
```

### 1.3 安装 Python SDK（在第 2 步建好的 conda 环境里装）
```bash
pip install ./airbot_py-5.1.6-py3-none-any.whl
python -c "import airbot_py; print('airbot_py OK')"
```

### 1.4 相机库
```bash
pip install pyrealsense2
```

> 机械臂封装 [airbot/play_sdk.py](airbot/play_sdk.py) 已包含在本仓库里，无需单独安装。

---

## 2. 本地环境（conda，方案 A）—— 采集 + 推理

> 一个 conda 环境装下 openpi + 机械臂 SDK + 相机，`collect_data.py` 和 `local_inference.py` 都在这里跑。

```bash
# 拉代码（务必带子模块）
git clone --recurse-submodules <本仓库地址> Pi0.5_reproduction
cd Pi0.5_reproduction

# 新建 conda 环境（python 3.11，openpi 要求 ≥3.11）
conda create -n openpi_airbot python=3.11 -y
conda activate openpi_airbot

# 安装 openpi 全家桶
pip install -e packages/openpi-client
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pip install -e .

# 安装机械臂 SDK + 相机（见第 1 章）
pip install ./airbot_py-5.1.6-py3-none-any.whl
pip install pyrealsense2
```

> 系统级 NVIDIA 驱动 / CUDA 需自行装好，`nvidia-smi` 能正常输出即可。

---

## 3. 服务器训练环境（uv）

> 训练服务器不接机械臂，只需 openpi。用 uv 安装。

```bash
# 安装 uv（装在 conda 之外，全局可用；不要 pip install 进 conda 环境）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 拉代码
git clone --recurse-submodules <本仓库地址> Pi0.5_reproduction
cd Pi0.5_reproduction

# uv 按 uv.lock 装好 openpi + jax + torch 等全部依赖到项目本地 .venv
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
后续训练命令统一用 `uv run python ...`。

---

## 4. 数据集 & 权重（HuggingFace）

数据集（约数 GB）和微调权重（约 9GB）**不进 git**，通过 HuggingFace 分发。

### 4.1 仓库地址（建好后填写）
- 数据集：`<待填：你的HF用户名>/airbot_play_data`
- 权重：  `<待填：你的HF用户名>/pi05_airbot_play`

### 4.2 上传（数据/权重的产出方执行）
```bash
huggingface-cli login          # 用 Write 权限的 token

# 数据集（整个 lerobot 文件夹）
huggingface-cli upload <用户名>/airbot_play_data "$HF_LEROBOT_HOME/airbot_play_data" . --repo-type dataset
# 权重（某个 step 的 checkpoint）
huggingface-cli upload <用户名>/pi05_airbot_play ./checkpoints/pi05_airbot_play/<exp-name>/<step> . --repo-type model
```

### 4.3 下载
```bash
# 服务器下载数据集到 HF_LEROBOT_HOME（见第 5.2 节）
huggingface-cli download <用户名>/airbot_play_data --repo-type dataset \
  --local-dir "$HF_LEROBOT_HOME/airbot_play_data"

# 本地下载权重用于推理
huggingface-cli download <用户名>/pi05_airbot_play --repo-type model \
  --local-dir ./checkpoints/pi05_airbot_play/my_experiment/16000
```

---

## 5. 完整流程

### 5.1 采集数据（本地，conda 环境）
确保机械臂上电、双臂 gRPC 服务已启动、两台相机已连接，然后：
```bash
conda activate openpi_airbot
python airbot/collect_data.py --task "pick up the block and place it in the bowl"
```
- 按 `Enter` 开始录一条 episode，遥操作 Lead 臂演示，再按 `Enter` 保存（`d` 丢弃），`q` 结束并自动存为 LeRobot 格式。
- 数据保存在 `$HF_LEROBOT_HOME/airbot_play_data`（默认 `~/.cache/huggingface/lerobot/`）。
- 采完按第 4.2 节上传到 HuggingFace。

### 5.2 训练（服务器，uv 环境）
```bash
# 1) 指定数据集根目录（放在大数据盘，避免塞满系统盘）
mkdir -p /path/to/bigdisk/data
export HF_LEROBOT_HOME=/path/to/bigdisk/data
#    然后把数据集放到 $HF_LEROBOT_HOME/airbot_play_data（下载或拷贝）

# 2) 修复 parquet 元数据兼容性（List -> Sequence）
uv run python scripts/repack_dataset.py        # 自动读取 HF_LEROBOT_HOME

# 3) 计算归一化统计量
uv run scripts/compute_norm_stats.py --config-name pi05_airbot_play

# 4) 训练（LoRA 微调，配置见 src/openpi/training/config.py 的 pi05_airbot_play）
uv run scripts/train.py pi05_airbot_play --exp-name=test_run --num-train-steps=30000 --batch-size=8
```
- 训练基础权重 `pi05_base` 会自动从 `gs://openpi-assets` 下载（需联网）。
- 输出权重在 `checkpoints/pi05_airbot_play/<exp-name>/<step>/`，含归一化统计 `assets/`。
- 训练完按第 4.2 节上传权重。

> `HF_LEROBOT_HOME` 是 lerobot 存放数据集的根目录。采集、repack、训练三处都按它定位数据集，所以**训练前必须先 export**。

### 5.3 推理（本地，conda 环境）
```bash
conda activate openpi_airbot

# 先 dry-run：只推理打印、不发运动指令（仍会连机械臂和相机）
python local_inference.py \
  --checkpoint ./checkpoints/pi05_airbot_play/my_experiment/16000 \
  --task "pick up the block and place it in the bowl" \
  --steps 5 --freq 5 --dry-run

# 确认预测正常后，去掉 --dry-run 真机运行
python local_inference.py \
  --checkpoint ./checkpoints/pi05_airbot_play/my_experiment/16000 \
  --task "pick up the block and place it in the bowl"
```
**安全**：有显示窗口时按 `q` / `Esc` 急停并回零位；无窗口时（`--no-display`）在终端输入 `q` 回车急停。`--min-z` 设末端最低高度防止压台面。

---

## 6. 测试后清理

```bash
conda deactivate
conda env remove -n openpi_airbot          # 删 conda 环境
rm -rf .venv                               # 删 uv 环境（如有）
# 模型缓存（跨项目共享，按需清理）：
du -sh ~/.cache/openpi ~/.cache/huggingface
```

---

## 仓库中本项目新增/修改的关键文件
| 路径 | 说明 |
|------|------|
| [airbot/collect_data.py](airbot/collect_data.py) | 遥操作 + 双相机采集 → LeRobot 数据集 |
| [airbot/play_sdk.py](airbot/play_sdk.py) | AIRBOT Play 机械臂 + 相机封装 |
| [airbot/play_config.json](airbot/play_config.json) | 端口 + 相机序列号配置 |
| [local_inference.py](local_inference.py) | 本地实时推理脚本 |
| `src/openpi/policies/airbot_play_policy.py` | airbot 数据 ↔ 模型输入输出变换 |
| `src/openpi/training/config.py` | 训练配置 `pi05_airbot_play` |
| `scripts/repack_dataset.py` | parquet 元数据兼容性修复 |
