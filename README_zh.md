# π0.5 在 AIRBOT Play 上的末端空间训练与部署

本仓库在 [openpi](https://github.com/Physical-Intelligence/openpi) 的 **π0.5** 上做微调,
训练数据是一条或两条 **AIRBOT Play** 机械臂的遥操作示教。状态和动作用末端位姿表示,而不是关节角。
仓库覆盖完整流程:数据采集、数据集打包、训练、真机部署。

本页说明模型是用什么数据训练的,以及如何复现这套训练。

> English version: [README.md](README.md)。上游 openpi 原始文档:[README_openpi.md](README_openpi.md)。

---

## 1. 模型用什么数据训练

### 1.1 原始录制数据

数据通过遥操作采集:人拖动主臂,从臂跟随。每条 episode 以 10 Hz 录制,每帧包含:

- RGB 图像,640×480:环境相机、臂 0 的手眼相机,以及(仅双臂)臂 1 的手眼相机;
- 机械臂自己报的从臂末端位姿(`get_end_pose`:基座系下的位置和四元数)和夹爪读数(`get_eef_pos`);
- 主臂的夹爪指令;
- 文本形式的任务指令。

模型的本体感知输入全部来自机械臂对自身的报告:末端位姿和夹爪开度。
关节角、关节速度和力矩都不是模型输入。

原始录制数据在磁盘上的确切格式,以及给其他机械臂写采集脚本时必须遵守的约定,见
[airbot/DATA_FORMAT.md](airbot/DATA_FORMAT.md)。

### 1.2 状态与动作:20 维

每条臂占 10 个数。臂 0 是右臂,臂 1 是左臂。

| 下标 | 内容 |
|---|---|
| 0–2 | 臂 0 位置,单位米 |
| 3–8 | 臂 0 姿态,6D 表示(旋转矩阵的前两列) |
| 9 | 臂 0 夹爪开度,0 = 闭合,1 = 全开 |
| 10–19 | 臂 1,排列相同 |

- **坐标系。** 位姿表达在 **W′** 系下。W′ 对每条 episode、每条臂各建一个:原点是首帧时机械臂报的末端点,
  z 轴就是基座的 z 轴(重力方向),并绕 z 轴转一个角度,使首帧的 yaw 为零。
  所以每条 episode 都从位置 (0, 0, 0) 开始。
- **参考点。** 位姿就是机械臂报的那个点,不加任何工具偏移(打包脚本的 `--arm-frame`,配置里的 `arm_frame=True`)。
- **夹爪。** 读数按 (读数 − 闭合值) / (全开值 − 闭合值) 归一化并截断到 [0, 1]。每把夹爪的这两个值存在
  [gripper_geom/gripper_params.py](gripper_geom/gripper_params.py) 里。
- **状态**:第 t 帧的状态是从臂实测的位姿和夹爪开度。
- **动作**:第 t 帧的动作分两部分。位姿部分是第 t+1 帧的状态,是 W′ 下的绝对目标,不是增量。
  夹爪部分用主臂的指令开度,因为实测开度碰到物体就停在物体的宽度上,表达不出"夹紧"。
- **单臂 episode**:臂 1 填固定的静止向量 `[0,0,0, 1,0,0, 0,1,0, 0]`,`arm1_mask = 0`,臂 1 手眼相机用全黑图像。

### 1.3 打包后的数据集

打包脚本输出一个 10 fps 的 LeRobot(v2.1)数据集。

| 列 | 形状 | 训练是否读取 | 含义 |
|---|---|---|---|
| `image` | 480×640×3,H.264 | 是 | 环境相机 |
| `wrist_image` | 480×640×3,H.264 | 是 | 臂 0 手眼相机 |
| `wrist_image_1` | 480×640×3,H.264 | 是 | 臂 1 手眼相机(单臂 episode 为全黑) |
| `state` | 20 | 是 | 见 1.2 |
| `actions` | 20 | 是 | 见 1.2 |
| `env_mask` | 1 | 是 | 环境相机有效时为 1 |
| `arm1_mask` | 1 | 是 | 双臂 episode 为 1,单臂为 0 |
| `gripper_id_0`、`gripper_id_1` | 各 1 | 是 | 每条臂上装的是哪把夹爪,取值是它在 `gripper_names` 里的下标 |
| `task_index` | 1 | 是 | 用来取文本指令 |
| `effort`、`effort_mask` | 14、1 | 否 | 遥测记录的电机力矩;本配置不读取 |
| `timestamp`、`frame_index`、`episode_index`、`index` | 各 1 | 仅作索引 | |

### 1.4 实际送进模型的内容

每个训练样本:

- **图像。** 三个视角带填充地缩放到 224×224。环境相机放进 `base_0_rgb` 槽,臂 0 手眼放进
  `left_wrist_0_rgb`,臂 1 手眼放进 `right_wrist_0_rgb`。(槽位名来自 openpi,和这里的左右并不对应。)
  `env_mask = 0` 时环境槽被屏蔽,`arm1_mask = 0` 时臂 1 的槽被屏蔽。
- **状态。** 20 维状态用它的第 1 和第 99 百分位做归一化,补零到 32 维;和标准 π0.5 一样,
  状态被离散化后写进文本 prompt。
- **指令。** 任务文本,用 PaliGemma 分词器分词(最多 200 个 token)。
- **夹爪点云。** 每条臂一份固定的夹爪点云,按 `gripper_id_0` / `gripper_id_1` 从
  `gripper_geom/grippers_armframe.npz` 里查表得到(见 4.2)。
- **训练目标。** 之后的 10 个动作(一秒),每个 20 维,用各自的第 1 和第 99 百分位归一化,补零到 32 维。

我们自己的数据集不在本仓库里。它包含约 10 小时示教,大约 700 条 episode,涵盖单臂和双臂的桌面任务。

---

## 2. 训练设置

配置名是 `pi05_cotrain_dualarm_armframe`,定义在 `src/openpi/training/config.py`。

| 项目 | 取值 |
|---|---|
| 初始权重 | π0.5 base,`gs://openpi-assets/checkpoints/pi05_base/params`(自动下载) |
| 微调方式 | 两个 Gemma 骨干都用 LoRA(`gemma_2b_lora`、`gemma_300m_lora`);它们的非 LoRA 权重由 openpi 的 LoRA 冻结规则冻结 |
| 动作块 | 10 步,10 Hz |
| Batch size | 48 |
| 训练步数 | 40,000 |
| 学习率 | 余弦衰减;预热 1,000 步,峰值 3.5e-5,终值 3.5e-6 |
| 优化器 | AdamW(β1 0.9,β2 0.95),梯度范数裁剪到 1.0 |
| EMA | 关闭 |
| 归一化 | 状态和动作都用分位数(第 1 / 第 99 百分位) |
| 随机种子 | 42 |
| 我们使用的硬件 | 两块 NVIDIA H200 |

---

## 3. 环境准备

### 3.1 两类机器、两套环境

| 机器 | 环境 | 用途 |
|---|---|---|
| 本地工作站(机械臂、相机、GPU) | conda,一个环境 | 采集、打包、部署 |
| 训练服务器(大显存 GPU,无机器人) | uv | 算 norm stats、训练 |

部署脚本要在同一个进程里加载模型和机械臂 SDK,所以本地用一个 conda 环境最省事。服务器只训练。

### 3.2 本套硬件

- **机械臂**:最多两对主从臂。臂 0(右):从臂 gRPC 端口 `50050`,主臂 `50051`。
  臂 1(左):从臂 `50052`,主臂 `50053`。
- **相机**:三个 RealSense D405。环境 `230422271972`,臂 0 手眼 `230422271433`,臂 1 手眼 `218622271178`。
  序列号写在 [airbot/collect_data.py](airbot/collect_data.py) 里,也是
  [local_inference_eef.py](local_inference_eef.py) 的参数默认值。
- **GPU**:openpi 的建议是推理 ≥ 8 GB,LoRA 微调 ≥ 22.5 GB。
- **系统**:Ubuntu 20.04 及以上。

相机和 USB-CAN 适配器要接在不同的 USB 控制器上。共用一条总线时,相机重新枚举可能让 CAN 掉线。

### 3.3 AIRBOT 驱动和 SDK(仅真机端)

AIRBOT 的专有软件不在本仓库里。需要:

```bash
sudo apt-get install ./airbot-configure_5.1.6-1_all.deb   # CAN / udev 驱动
pip install ./airbot_py-5.1.6-py3-none-any.whl            # Python SDK, 装进下面的 conda 环境
pip install pyrealsense2
```

### 3.4 本地环境(conda)

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
conda create -n openpi_airbot python=3.11 -y && conda activate openpi_airbot
pip install -e packages/openpi-client
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pip install -e .
pip install ./airbot_py-5.1.6-py3-none-any.whl pyrealsense2
```

### 3.5 服务器环境(uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules <repo-url> && cd <repo>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

服务器需要 `ffmpeg` 来解码 H.264 相机视频。

---

## 4. 分步操作

下文的 `$DATA` 是存放原始录制数据和打包数据集的目录。

### 4.1 采集

```bash
# 单臂(臂 0); 装的是哪把夹爪就写哪把
python airbot/collect_data.py --task "hang the mug on the mug tree" \
  --follow-gripper get --repo-id get_mug_tree --output-dir $DATA/GET/mug_tree_task

# 双臂
python airbot/collect_data.py --dual-arm --task "sweep the objects into the dustpan" \
  --arm0-gripper get --arm1-gripper parallel \
  --repo-id sweep_objects --output-dir $DATA/GET/sweep_task
```

夹爪名就是 `gripper_geom/gripper_params.py` 里的键(`parallel` 是出厂夹爪,`get` 是另一把)。
夹爪名决定了把主臂夹爪映射到从臂夹爪时用的开合范围。

`--task` 的字符串就是训练时的指令,部署时要用完全相同的字符串。状态里没有关节角,
所以同一个任务的所有 episode 要从相同的机械臂构型开始。

脚本还会给每条 episode 写一个 100 Hz 的遥测文件(本训练不使用;`--telemetry-hz 0` 可关闭),
并在关节接近限位时在终端提示。

### 4.2 夹爪相关文件

训练和部署对每把夹爪需要两样东西:

- **开合范围**:[gripper_geom/gripper_params.py](gripper_geom/gripper_params.py) 里的 `close` 和 `open`,
  它们是 `get_eef_pos` 的读数。`airbot/teach_probe.py` 可以一步步开合夹爪并打印读数。
- **点云**:`gripper_geom/grippers_armframe.npz`,每个夹爪名对应一份。
  它由 `python gripper_geom/align_to_arm_frame.py` 从 `gripper_geom/grippers.npz` 生成;
  后者由各夹爪的描述符合并而成,描述符用 `gripper_geom/build_gripper_descriptor.py`(CAD 网格)和
  `build_parallel_descriptor.py`(URDF)生成。合并这一步在本仓库里没有脚本。

`.npz` 是生成文件,不进 git。本地和服务器上都要在相同路径放好。夹爪名的顺序在打包脚本的
`--gripper-names` 和配置的 `gripper_names` 里必须一致(默认 `parallel get`)。

### 4.3 打包

```bash
python umi/pack_lerobot_fast.py --arm-frame \
  --template $DATA/_schema_template \
  --teleop-dir     $DATA/parallel/mug_tree_task/parallel_mug_tree  $DATA/GET/mug_tree_task/get_mug_tree \
  --teleop-gripper parallel                                       get \
  --dualarm-dir     $DATA/GET/sweep_task/sweep_objects \
  --dualarm-gripper get parallel \
  --repo-id cotrain_dualarm5 -o $DATA/cotrain_dualarm5 --verify
```

- `--teleop-dir` 列出单臂录制目录,`--teleop-gripper` 给每个目录配一个夹爪名。`--dualarm-dir` 列出双臂录制目录,
  `--dualarm-gripper` 给的是(臂 0,臂 1)一对夹爪名:只给一对则用于所有目录,也可以每个目录给一对。
- `--arm-frame` 把位姿写成机械臂报的那个点(见 1.2)。它必须和配置里的 `arm_frame=True` 配套使用。
- `--template` 是一个已有的 20 维打包数据集,只读取它的 schema。
- 视频直接复制,不重新编码。

打包脚本会在 `$DATA/_pack_ledger/<repo-id>/` 里留一份很小的账本,所以打包产物上传后可以删掉。
之后追加新录制的数据:

```bash
python umi/pack_lerobot_fast.py --arm-frame --append \
  --teleop-dir $DATA/<新录制目录> --teleop-gripper <夹爪名> \
  --repo-id cotrain_dualarm5 -o $DATA/cotrain_dualarm5
```

这里必须写 `--repo-id`,脚本靠它找到账本。如果打包产物已经删了,这条命令只写出新增的 episode
和完整的元数据,用 `rsync`(不带 `--delete`)合并进服务器上的那份即可。

### 4.4 Norm stats 与训练(服务器)

把打包好的数据集拷到服务器,然后:

```bash
export HF_LEROBOT_HOME=/path/to/packed/datasets      # 包含 cotrain_dualarm5/ 的那个目录
uv run scripts/compute_norm_stats.py --config-name pi05_cotrain_dualarm_armframe
uv run scripts/train.py pi05_cotrain_dualarm_armframe --exp-name=<run> --overwrite \
  --save-interval=2000 --keep-period=None
```

- 配置按它的 `repo_id`(`cotrain_dualarm5`)读取数据集。如果打包时用了别的名字,到 `config.py` 里改。
- 数据集一变就要重算 norm stats。
- 权重保存在 `checkpoints/<配置名>/<run>/<步数>/`。按上面的参数,磁盘上只保留最新一档,最后一步一定会保存。
  要续训被中断的训练,把 `--overwrite` 换成 `--resume`。
- [train_grip_fast.sh](train_grip_fast.sh) 是我们在可抢占队列上用的 SLURM 脚本:等数据集和 norm stats
  就绪后开始训练或续训。里面的路径是我们集群专用的。

### 4.5 部署(本地)

把权重目录(`params/` 和 `assets/`)拷到工作站,然后:

```bash
python local_inference_eef.py \
  --checkpoint checkpoints/pi05_cotrain_dualarm_armframe/<run>/<步数> \
  --config-name pi05_cotrain_dualarm_armframe \
  --task "hang the mug on the mug tree" \
  --arm0-gripper get --start-from-lead --dry-run
```

打印出来的目标位姿看着没问题后,去掉 `--dry-run`。

| 参数 | 作用 |
|---|---|
| `--dual-arm --arm1-gripper <夹爪名>` | 驱动两条臂 |
| `--start-from-lead` | 先让从臂对齐到主臂的关节角,保证 rollout 开局的机械臂构型和示教一致 |
| `--chunk N` | 每 N 步重新推理一次(默认 10:整段执行完再推理) |
| `--timeout S` | S 秒后停止 |
| `--log-joints` | 打印关节角和关节限位提示 |

按 `q` 或 `Esc` 停止。退出时机械臂停在原地并张开夹爪,再按回车才回到零位。
`--config-name` 必须是训练该权重时用的配置。

---

## 5. 关键文件

| 路径 | 用途 |
|---|---|
| [airbot/collect_data.py](airbot/collect_data.py) | 遥操作采集(单臂 / 双臂,AIRBOT Play) |
| [airbot/collect_data_yam.py](airbot/collect_data_yam.py) | 同上,YAM 臂 + GELLO 主臂 |
| [airbot/DATA_FORMAT.md](airbot/DATA_FORMAT.md) | 原始采集格式与新臂接入约定 |
| [airbot/play_sdk.py](airbot/play_sdk.py) | 部署时用的 AIRBOT Play 封装 |
| [gripper_geom/gripper_params.py](gripper_geom/gripper_params.py) | 每把夹爪的开合范围 |
| [gripper_geom/](gripper_geom/) | 生成夹爪点云文件的脚本 |
| [umi/pack_lerobot_fast.py](umi/pack_lerobot_fast.py) | 打包:原始录制 → 一个 20 维数据集,支持增量追加 |
| `src/openpi/training/config.py` | 训练配置 `pi05_cotrain_dualarm_armframe` |
| `src/openpi/policies/airbot_eef_policy.py` | 数据集各列与模型输入之间的映射 |
| `scripts/compute_norm_stats.py`、`scripts/train.py` | 算 norm stats、训练 |
| [local_inference_eef.py](local_inference_eef.py) | 真机部署(单臂 / 双臂) |
| [train_grip_fast.sh](train_grip_fast.sh) | SLURM 脚本(集群专用) |

---

## 6. 说明

- 数据集和权重不进 git。
- `local_inference.py`、`eval_eef_offline.py` 和 `umi/pack_lerobot.py` 属于更早的流程(关节空间,以及 10 维单臂),
  它们的训练配置已经删除,在本文描述的配置下跑不起来。

## 许可与致谢

本仓库基于 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi),
沿用同样的 Apache 2.0 许可([LICENSE](LICENSE))。这不是 openpi 的官方发布。
