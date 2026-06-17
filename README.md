# AirBot Play × openpi (π0.5): Reproduction & Deployment Guide

This repo reproduces the **π0.5 (pi05)** vision-language-action model from
[openpi](https://github.com/Physical-Intelligence/openpi) and adapts it to the
**AIRBOT Play** arm. It covers two end-to-end pipelines:

- **Pipeline A — joint space, single task**: teleop collection → server training → local real-time inference (`local_inference.py`, config `pi05_airbot_play`).
- **Pipeline B — task space (EEF), UMI + teleop co-training**: convert handheld **UMI** data and AIRBOT teleop data into one shared task-space dataset → co-train → deploy (`local_inference_eef.py`, config `pi05_cotrain_eef`). This is the current focus: injecting new skills (e.g. *wipe the blackboard*, collected cheaply with a handheld UMI gripper) into a robot that is also trained on teleop tasks (e.g. *pick up the block*).

> 中文版见 [README_zh.md](README_zh.md). Upstream openpi docs: [README_openpi.md](README_openpi.md).

---

## 0. Overview

### Two machines, two environments
| Machine | Env tool | Used for |
|---|---|---|
| **Local workstation** (arm + cameras + GPU) | **conda** (one env for everything) | data collection, local inference |
| **Training server** (big GPU, no arm) | **uv** | training, norm-stats |

The local inference scripts load the openpi model **and** the arm SDK in the
same process, so one conda env is simplest. The server only trains, so it uses
openpi's recommended `uv`.

### Hardware (this setup; change the configs if yours differs)
- **Arms ×2**: Lead (Replay, passive leader, gRPC port `50051`) + Follow (Play, powered follower, port `50050`).
- **Cameras ×2**: RealSense D405 — head/env SN `230422271972`, wrist/eye-in-hand SN `230422271433`.
- **GPU**: inference ≥8 GB; LoRA fine-tuning ≥22.5 GB.
- **OS**: Ubuntu 20.04+.

Camera serials live in [airbot/play_config.json](airbot/play_config.json) and [airbot/collect_data.py](airbot/collect_data.py).

---

## 1. Arm & camera setup (system level, real-robot side only)

Proprietary AIRBOT software is **not** in this repo. For Ubuntu + pure-Python use you only need:

| File | Purpose |
|---|---|
| `airbot-configure_5.1.6-1_all.deb` | driver (CAN / udev) |
| `airbot_py-5.1.6-py3-none-any.whl` | Python SDK |

```bash
sudo apt-get install ./airbot-configure_5.1.6-1_all.deb   # driver
pip install ./airbot_py-5.1.6-py3-none-any.whl            # SDK (into the conda env from §2)
pip install pyrealsense2                                  # cameras
```

The arm wrapper [airbot/play_sdk.py](airbot/play_sdk.py) is already in the repo.

---

## 2. Local environment (conda) — collection + inference

```bash
git clone --recurse-submodules <repo-url> && cd <repo>
conda create -n openpi_airbot python=3.11 -y
conda activate openpi_airbot

pip install -e packages/openpi-client
pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
pip install -e .
pip install ./airbot_py-5.1.6-py3-none-any.whl pyrealsense2
```

System NVIDIA driver / CUDA must already work (`nvidia-smi`).

---

## 3. Server environment (uv) — training

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --recurse-submodules <repo-url> && cd <repo>
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
All training commands then use `uv run python ...`.

> The server needs `ffmpeg` for video decoding (`apt-get install ffmpeg`); the
> LeRobot datasets here store cameras as H.264 video.

---

## 4. Pipeline A — joint space, single task

State/action = 6 joints + gripper (7D), absolute joint targets.

```bash
# Collect (local)
python airbot/collect_data.py --task "pick up the block and place it in the bowl"

# Train (server)
export HF_LEROBOT_HOME=/path/to/bigdisk/data
uv run scripts/compute_norm_stats.py --config-name pi05_airbot_play
uv run scripts/train.py pi05_airbot_play --exp-name=run1 --num-train-steps=30000 --batch-size=8

# Infer (local) — dry-run first
python local_inference.py --checkpoint ./checkpoints/pi05_airbot_play/run1/30000 \
  --task "pick up the block and place it in the bowl" --dry-run
```
Safety: `q`/`Esc` (or `q`+Enter with `--no-display`) stops and returns to zero;
`--min-z` caps the lowest end-effector height.

---

## 5. Pipeline B — task space (EEF), UMI + teleop co-training

The goal: collect a new skill cheaply with a **handheld UMI gripper** and inject
it into the robot, co-trained with AIRBOT teleop data, by mapping both sources
into **one shared task-space representation**.

### 5.1 Shared representation
Every sample (UMI and teleop) is converted to the same schema:

- **state / action = 10D = `pos(3) + rot6d(6) + gripper(1)`**, expressed in a
  per-episode relative frame **W′** (origin = first-frame fingertip, z = gravity,
  first-frame yaw zeroed; pitch/roll kept gravity-absolute). This makes both
  sources "rootless" and embodiment-agnostic.
- **gripper** normalized by the robot's max opening (÷0.073, clip [0,1]); 0 = closed.
  The action channel uses the **lead/commanded** gripper so grasps express
  closing force (not just the achieved width).
- **two camera slots**: `wrist_image` (always valid) and `image` (env/head camera).
- **`env_mask`**: 1 = env camera valid (teleop), 0 = masked (UMI has no global
  camera). This lets a single policy co-train across heterogeneous sensor sets.
- **camera FOV match**: UMI's fisheye is de-fisheyed to ~79° to match the D405's 78.6°.

### 5.2 Steps
```bash
# (a) Collect teleop EEF data — records base-frame EEF (pos+quat+gripper) via the
#     vendor FK get_end_pose. The lead→follow gripper-range correction is built in.
python airbot/collect_data.py --task "pick up the block and place it in the bowl" \
  --output-dir /home/you/pi_data --repo-id airbot_play_data
python airbot/collect_data.py --task "wipe the blackboard" \
  --output-dir /home/you/pi_data --repo-id airbot_wipe_data

# (b) Pack UMI mcaps + (one or more) teleop datasets into one EEF dataset.
#     Each teleop episode's prompt is read automatically from its source tasks.jsonl;
#     datasets-v4 "List" feature type is auto-downgraded for server compatibility.
python umi/pack_lerobot.py \
  --umi-dir /path/to/umi_mcaps \
  --teleop-dir /home/you/pi_data/airbot_play_data /home/you/pi_data/airbot_wipe_data \
  -o /home/you/pi_data/cotrain_eef --repo-id cotrain_eef --verify
#   (--skip-umi packs teleop only, for the ablation config pi05_teleop_eef)

# (c) Train (server)
uv run scripts/compute_norm_stats.py --config-name pi05_cotrain_eef
uv run scripts/train.py pi05_cotrain_eef --exp-name cotrain_v1

# (d) Deploy (local). env-mask=1 uses the head camera (recommended once the task
#     has teleop data); single-camera UMI-only tasks use env-mask=0.
python local_inference_eef.py --checkpoint ./checkpoints/pi05_cotrain_eef/cotrain_v1/17000 \
  --task "pick up the block and place it in the bowl" --env-mask 1 \
  --speed-profile fast --speed-scale 0.3
```

### 5.3 Offline open-loop eval (no robot)
Validate that the model actually learned an episode by feeding recorded
observations and comparing predicted vs ground-truth actions (a low flow-matching
loss does **not** guarantee good sampled actions):
```bash
python eval_eef_offline.py --checkpoint <ckpt> --dataset /home/you/pi_data/cotrain_eef \
  --task-filter "block" --plot eval_block.png
#  --force-env-mask 0/1 to probe sensitivity to the env-camera mask
```

> Deployment is closed-loop: the model is re-queried each chunk from the live
> `get_end_pose`, which is self-correcting. Note that a wrongly-actuated gripper
> dimension feeds back into the observed state and corrupts pose prediction too —
> so the gripper denormalization and lead-gripper action source matter for the
> whole policy, not just the gripper.

---

## 6. Datasets & weights (HuggingFace)

Datasets and fine-tuned weights are **not** in git.
```bash
hf auth login
hf upload <user>/<dataset> "$HF_LEROBOT_HOME/<dataset>" . --repo-type dataset
cd checkpoints/<config>/<exp>/<step> && hf upload <user>/<weights> . --exclude "train_state/*"
```
`HF_LEROBOT_HOME` is the LeRobot dataset root; collection, packing and training
all resolve datasets relative to it, so export it before training.

---

## 7. Key files
| Path | Purpose |
|---|---|
| [airbot/collect_data.py](airbot/collect_data.py) | Teleop + dual-camera collection → LeRobot (joint 7D **and** base-frame EEF 8D); lead→follow gripper rescaling. |
| [airbot/play_sdk.py](airbot/play_sdk.py) | AIRBOT Play arm + camera wrapper. |
| [airbot/play_config.json](airbot/play_config.json) | Ports, camera serials, workspace bounds. |
| [umi/umi_to_lerobot.py](umi/umi_to_lerobot.py) | UMI mcap → de-fisheyed video + W′ pose + gripper-norm. |
| [umi/pack_lerobot.py](umi/pack_lerobot.py) | Stage-4 packer: UMI + teleop → one shared EEF dataset. |
| [umi/replay_check.py](umi/replay_check.py) | Real-robot replay validation of the coordinate chain. |
| [local_inference.py](local_inference.py) | Joint-space real-time inference (Pipeline A). |
| [local_inference_eef.py](local_inference_eef.py) | Task-space (EEF) real-time inference (Pipeline B). |
| [eval_eef_offline.py](eval_eef_offline.py) | Offline open-loop eval (predicted vs GT actions). |
| `src/openpi/policies/airbot_eef_policy.py` | EEF data ↔ model transforms (10D state, env_mask, optional gripper point cloud). |
| `src/openpi/training/config.py` | Configs `pi05_airbot_play`, `pi05_cotrain_eef`, `pi05_teleop_eef`, `pi05_teleop_eef_grip`. |
| [gripper_geom/](gripper_geom/) | Gripper-geometry toolkit (see §6). |

---

## 8. Roadmap — gripper-aware, cross-gripper training (in progress)

The current research direction: **teach the policy to exploit a 1-DOF gripper's
distinct functional regions** (e.g. the GET gripper's fingertip vs. rear) **and to
co-train across heterogeneous grippers** (parallel / GET / Robotiq) in one model.

Method (Option C): encode each gripper's CAD-sampled **point cloud (in the TCP /
`get_end_pose` frame)** into one token via a small PointNet, splice it into the pi0
prefix (`embed_prefix`). The action expert attends to it, so the policy can compose
"where the TCP moves" with "where each region is" → uses the right region per task,
and adapts when the gripper (token) changes. The geometry is **declarative & constant
per gripper**, so existing datasets are retrofitted by tagging — no re-collection.

Toolkit in [gripper_geom/](gripper_geom/):
- `inspect_meshes.py` — confirm which URDF meshes are the gripper.
- `build_parallel_descriptor.py` — assemble gripper meshes into the TCP frame via the
  URDF joints, sample a point cloud + functional-region anchors, save a descriptor
  `.npy`; `--tcp-rpy` aligns every gripper to one common convention (+X = approach, +Z = up).
- `project_gripper_overlay.py` — project the cloud onto the live wrist camera (coarse sanity check).
- `axis_check.py` — kinematically verify the `get_end_pose` tool-frame axes.

Network changes (behind `Pi0Config.gripper_token`, default off → no behavior change):
`gripper_pc` field on `Observation`, a `PointNetEncoder` in `pi0.py`, the prefix splice,
and the descriptor injection in `AirbotEEFInputs`. Config `pi05_teleop_eef_grip` enables
it for a parallel-only **regression A/B** (token on vs off should match on the
gold-standard teleop data before mixing in other grippers).

> Descriptor `.npy` files are gitignored (regenerable from CAD); copy them to the
> server before training a `gripper_token` config.

---

## 9. Cleanup
```bash
conda env remove -n openpi_airbot
rm -rf .venv
du -sh ~/.cache/openpi ~/.cache/huggingface   # shared caches, clear as needed
```
