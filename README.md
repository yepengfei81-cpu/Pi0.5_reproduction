# π0.5 on AIRBOT Play: End-Effector-Space Training and Deployment

This repository fine-tunes [openpi](https://github.com/Physical-Intelligence/openpi)'s
**π0.5** on teleoperated demonstrations from one or two **AIRBOT Play** arms.
States and actions are end-effector poses rather than joint angles. The
repository covers the whole path: data collection, dataset packing, training
and real-robot deployment.

This page describes what the model is trained on and how to reproduce the
training.

> 中文版: [README_zh.md](README_zh.md). Upstream openpi documentation: [README_openpi.md](README_openpi.md).

---

## 1. What the model is trained on

### 1.1 Raw recordings

Data is collected by teleoperation: a person moves a leader arm and a follower
arm tracks it. Each episode is recorded at 10 Hz and holds, per frame:

- RGB images, 640×480: an environment camera, the wrist camera of arm 0, and
  (dual-arm only) the wrist camera of arm 1;
- the follower's end-effector pose as reported by the arm (`get_end_pose`:
  position and quaternion in the arm's base frame) and its gripper reading
  (`get_eef_pos`);
- the leader's gripper command;
- the task instruction as text.

The model's proprioceptive input comes entirely from what the arm reports about
itself: end-effector pose and gripper opening. Joint angles, velocities and
torques are not model inputs.

### 1.2 State and action: 20-D

Each arm contributes 10 numbers. Arm 0 is the right arm, arm 1 the left.

| Index | Content |
|---|---|
| 0–2 | arm 0 position, metres |
| 3–8 | arm 0 orientation, 6-D (first two columns of the rotation matrix) |
| 9 | arm 0 gripper opening, 0 = closed, 1 = fully open |
| 10–19 | arm 1, same layout |

- **Frame.** Poses are expressed in **W′**, a frame fixed per episode and per
  arm: its origin is the arm's reported end-effector point at the first frame,
  its z axis is the base z axis (gravity), and it is rotated about z so that the
  first frame's yaw is zero. Every episode therefore starts at position
  (0, 0, 0).
- **Reference point.** The pose is the point the arm reports, with no tool
  offset added (`--arm-frame` in the packer, `arm_frame=True` in the config).
- **Gripper.** The reading is normalised as (reading − closed) / (open − closed)
  and clipped to [0, 1], using the two values stored per gripper in
  [gripper_geom/gripper_params.py](gripper_geom/gripper_params.py).
- **State** at frame t is the follower's measured pose and gripper opening.
- **Action** at frame t has two parts. The pose part is the state at frame t+1,
  an absolute target in W′, not a delta. The gripper part is the leader's
  commanded opening, because the measured opening stops at the object's width
  and cannot express squeezing.
- **Single-arm episodes** fill arm 1 with a constant rest vector
  `[0,0,0, 1,0,0, 0,1,0, 0]`, set `arm1_mask = 0` and use a black image for the
  arm-1 wrist camera.

### 1.3 Packed dataset

The packer writes one LeRobot (v2.1) dataset at 10 fps.

| Column | Shape | Read by training | Meaning |
|---|---|---|---|
| `image` | 480×640×3, H.264 | yes | environment camera |
| `wrist_image` | 480×640×3, H.264 | yes | arm 0 wrist camera |
| `wrist_image_1` | 480×640×3, H.264 | yes | arm 1 wrist camera (black in single-arm episodes) |
| `state` | 20 | yes | see 1.2 |
| `actions` | 20 | yes | see 1.2 |
| `env_mask` | 1 | yes | 1 if the environment camera is valid |
| `arm1_mask` | 1 | yes | 1 for dual-arm episodes, 0 for single-arm |
| `gripper_id_0`, `gripper_id_1` | 1 each | yes | which gripper is mounted on each arm, as an index into `gripper_names` |
| `task_index` | 1 | yes | selects the text instruction |
| `effort`, `effort_mask` | 14, 1 | no | motor torque from the telemetry log; this config does not read it |
| `timestamp`, `frame_index`, `episode_index`, `index` | 1 each | bookkeeping | |

### 1.4 What reaches the model

For every training sample:

- **Images.** The three views are resized with padding to 224×224. The
  environment camera goes to the `base_0_rgb` slot, the arm-0 wrist camera to
  `left_wrist_0_rgb` and the arm-1 wrist camera to `right_wrist_0_rgb`. (The
  slot names come from openpi and do not match left and right here.) The
  environment slot is masked when `env_mask = 0`, the arm-1 slot when
  `arm1_mask = 0`.
- **State.** The 20-D state is normalised with its 1st and 99th percentiles,
  padded to 32 dimensions and, as in standard π0.5, discretised and written into
  the text prompt.
- **Instruction.** The task text, tokenised with the PaliGemma tokenizer
  (at most 200 tokens).
- **Gripper point cloud.** For each arm, a fixed point cloud of the mounted
  gripper, looked up by `gripper_id_0` / `gripper_id_1` in
  `gripper_geom/grippers_armframe.npz` (see 4.2).
- **Target.** The next 10 actions (one second), 20-D each, normalised with their
  1st and 99th percentiles and padded to 32 dimensions.

Our own dataset is not part of this repository. It holds about 10 hours of
demonstrations, roughly 700 episodes, covering single-arm and dual-arm tabletop
tasks.

---

## 2. Training settings

The config is `pi05_cotrain_dualarm_armframe` in `src/openpi/training/config.py`.

| Setting | Value |
|---|---|
| Base weights | π0.5 base, `gs://openpi-assets/checkpoints/pi05_base/params` (downloaded automatically) |
| Fine-tuning | LoRA on both Gemma backbones (`gemma_2b_lora`, `gemma_300m_lora`); their non-LoRA weights are frozen by openpi's LoRA freeze filter |
| Action chunk | 10 steps at 10 Hz |
| Batch size | 48 |
| Steps | 40,000 |
| Learning rate | cosine decay; 1,000 warm-up steps, peak 3.5e-5, final 3.5e-6 |
| Optimiser | AdamW (β1 0.9, β2 0.95), gradient clipping at norm 1.0 |
| EMA | off |
| Normalisation | quantile (1st / 99th percentile) for state and actions |
| Seed | 42 |
| Hardware we use | two NVIDIA H200 GPUs |

---

## 3. Setup

### 3.1 Two machines, two environments

| Machine | Environment | Used for |
|---|---|---|
| Local workstation (arms, cameras, GPU) | conda, one environment | collection, packing, deployment |
| Training server (large GPUs, no robot) | uv | norm stats, training |

The deployment script loads the model and the arm SDK in one process, so one
conda environment is simplest locally. The server only trains.

### 3.2 Hardware used here

- **Arms**: up to two leader-follower pairs. Arm 0 (right): follower on gRPC port
  `50050`, leader on `50051`. Arm 1 (left): follower `50052`, leader `50053`.
- **Cameras**: three RealSense D405. Environment `230422271972`, arm-0 wrist
  `230422271433`, arm-1 wrist `218622271178`. The serials are set in
  [airbot/collect_data.py](airbot/collect_data.py) and are the defaults of
  [local_inference_eef.py](local_inference_eef.py).
- **GPU**: openpi's guidance is ≥ 8 GB for inference and ≥ 22.5 GB for LoRA
  fine-tuning.
- **OS**: Ubuntu 20.04 or later.

Keep the cameras and the USB-CAN adapters on separate USB controllers. Camera
re-enumeration on a shared bus can drop the CAN link.

### 3.3 AIRBOT driver and SDK (robot side only)

The proprietary AIRBOT software is not in this repository. You need:

```bash
sudo apt-get install ./airbot-configure_5.1.6-1_all.deb   # CAN / udev driver
pip install ./airbot_py-5.1.6-py3-none-any.whl            # Python SDK, into the conda env below
pip install pyrealsense2
```

### 3.4 Local environment (conda)

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
conda create -n openpi_airbot python=3.11 -y && conda activate openpi_airbot
pip install -e packages/openpi-client
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pip install -e .
pip install ./airbot_py-5.1.6-py3-none-any.whl pyrealsense2
```

### 3.5 Server environment (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules <repo-url> && cd <repo>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

The server needs `ffmpeg` to decode the H.264 camera videos.

---

## 4. Step by step

Below, `$DATA` is the directory that holds raw recordings and packed datasets.

### 4.1 Collect

```bash
# single arm (arm 0); name the gripper that is mounted
python airbot/collect_data.py --task "hang the mug on the mug tree" \
  --follow-gripper get --repo-id get_mug_tree --output-dir $DATA/GET/mug_tree_task

# dual arm
python airbot/collect_data.py --dual-arm --task "sweep the objects into the dustpan" \
  --arm0-gripper get --arm1-gripper parallel \
  --repo-id sweep_objects --output-dir $DATA/GET/sweep_task
```

Gripper names are the keys of `gripper_geom/gripper_params.py` (`parallel` is
the factory gripper, `get` a second one). The name selects the opening range
used to map the leader's gripper to the follower's.

The `--task` string becomes the training instruction; use the same string at
deployment. Because the state holds no joint angles, start all episodes of a
task from the same arm configuration.

The script also writes a 100 Hz telemetry file per episode (not used by this
training; `--telemetry-hz 0` turns it off) and warns in the terminal when a
joint nears its limit.

### 4.2 Gripper files

Training and deployment need two things per gripper:

- **Opening range**: the `close` and `open` values in
  [gripper_geom/gripper_params.py](gripper_geom/gripper_params.py), which are
  `get_eef_pos` readings. `airbot/teach_probe.py` steps the gripper and prints
  the reading.
- **Point cloud**: `gripper_geom/grippers_armframe.npz`, one entry per gripper
  name. It is written by
  `python gripper_geom/align_to_arm_frame.py` from `gripper_geom/grippers.npz`,
  which is assembled from the per-gripper descriptors that
  `gripper_geom/build_gripper_descriptor.py` (CAD meshes) and
  `build_parallel_descriptor.py` (URDF) produce. The assembly step is not
  scripted in this repository.

The `.npz` files are generated and not in git. They must be present at the same
path locally and on the server. The order of names must be the same in the
packer's `--gripper-names` and the config's `gripper_names` (default:
`parallel get`).

### 4.3 Pack

```bash
python umi/pack_lerobot_fast.py --arm-frame \
  --template $DATA/_schema_template \
  --teleop-dir     $DATA/parallel/mug_tree_task/parallel_mug_tree  $DATA/GET/mug_tree_task/get_mug_tree \
  --teleop-gripper parallel                                       get \
  --dualarm-dir     $DATA/GET/sweep_task/sweep_objects \
  --dualarm-gripper get parallel \
  --repo-id cotrain_dualarm5 -o $DATA/cotrain_dualarm5 --verify
```

- `--teleop-dir` lists single-arm recordings and `--teleop-gripper` gives one
  gripper name per directory. `--dualarm-dir` lists dual-arm recordings and
  `--dualarm-gripper` gives an (arm 0, arm 1) pair, either one pair for all
  directories or one pair per directory.
- `--arm-frame` writes the pose at the arm's reported point (see 1.2). It must
  be used together with `arm_frame=True` in the config.
- `--template` is an existing packed 20-D dataset; only its schema is read.
- Videos are copied without re-encoding.

The packer keeps a small ledger in `$DATA/_pack_ledger/<repo-id>/`, so the
packed output can be deleted after it is uploaded. To add recordings later:

```bash
python umi/pack_lerobot_fast.py --arm-frame --append \
  --teleop-dir $DATA/<new recording> --teleop-gripper <name> \
  --repo-id cotrain_dualarm5 -o $DATA/cotrain_dualarm5
```

`--repo-id` is required here because it locates the ledger. If the packed
output was deleted, this writes only the new episodes plus complete metadata;
merge them into the server copy with `rsync` (without `--delete`).

### 4.4 Norm stats and training (server)

Copy the packed dataset to the server, then:

```bash
export HF_LEROBOT_HOME=/path/to/packed/datasets      # the directory containing cotrain_dualarm5/
uv run scripts/compute_norm_stats.py --config-name pi05_cotrain_dualarm_armframe
uv run scripts/train.py pi05_cotrain_dualarm_armframe --exp-name=<run> --overwrite \
  --save-interval=2000 --keep-period=None
```

- The config reads the dataset named by its `repo_id` (`cotrain_dualarm5`).
  Change it in `config.py` if you pack under another name.
- Recompute the norm stats whenever the dataset changes.
- Checkpoints go to `checkpoints/<config>/<run>/<step>/`. With the flags above
  only the latest one is kept, and the final step is always saved. Use
  `--resume` instead of `--overwrite` to continue an interrupted run.
- [train_grip_fast.sh](train_grip_fast.sh) is our SLURM script for a preemptible
  queue: it waits for the dataset and norm stats, then starts or resumes. Its
  paths are specific to our cluster.

### 4.5 Deploy (local)

Copy the checkpoint directory (`params/` and `assets/`) to the workstation, then:

```bash
python local_inference_eef.py \
  --checkpoint checkpoints/pi05_cotrain_dualarm_armframe/<run>/<step> \
  --config-name pi05_cotrain_dualarm_armframe \
  --task "hang the mug on the mug tree" \
  --arm0-gripper get --start-from-lead --dry-run
```

Remove `--dry-run` once the printed targets look right.

| Flag | Effect |
|---|---|
| `--dual-arm --arm1-gripper <name>` | drive both arms |
| `--start-from-lead` | first move the follower to the leader's joint angles, so the rollout starts in the same arm configuration as the demonstrations |
| `--chunk N` | re-plan every N steps (default 10: execute the whole chunk) |
| `--timeout S` | stop after S seconds |
| `--log-joints` | print joint angles and joint-limit warnings |

`q` or `Esc` stops the rollout. On exit the arm holds its position and opens
the gripper; pressing Enter then moves it to the zero position. `--config-name`
must be the config the checkpoint was trained with.

---

## 5. Key files

| Path | Purpose |
|---|---|
| [airbot/collect_data.py](airbot/collect_data.py) | Teleoperated collection, single or dual arm |
| [airbot/play_sdk.py](airbot/play_sdk.py) | AIRBOT Play wrapper used at deployment |
| [gripper_geom/gripper_params.py](gripper_geom/gripper_params.py) | Per-gripper opening range |
| [gripper_geom/](gripper_geom/) | Scripts that build the gripper point-cloud files |
| [umi/pack_lerobot_fast.py](umi/pack_lerobot_fast.py) | Packer: raw recordings → one 20-D dataset, with incremental append |
| `src/openpi/training/config.py` | Training config `pi05_cotrain_dualarm_armframe` |
| `src/openpi/policies/airbot_eef_policy.py` | Mapping between dataset columns and model inputs |
| `scripts/compute_norm_stats.py`, `scripts/train.py` | Norm stats and training |
| [local_inference_eef.py](local_inference_eef.py) | Real-robot deployment, single or dual arm |
| [train_grip_fast.sh](train_grip_fast.sh) | SLURM script (cluster-specific) |

---

## 6. Notes

- Datasets and checkpoints are not in git.
- `local_inference.py`, `eval_eef_offline.py` and `umi/pack_lerobot.py` belong to
  earlier pipelines (joint-space, and 10-D single-arm) whose training configs
  have been removed. They do not run with the config described here.

## License and acknowledgements

Built on [openpi](https://github.com/Physical-Intelligence/openpi) by Physical
Intelligence and released under the same Apache 2.0 license ([LICENSE](LICENSE)).
This is not an official openpi release.
