"""在服务器上重新打包 LeRobot 数据集，确保与 openpi 的 datasets 版本兼容"""
import shutil
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import json
import numpy as np
from PIL import Image

SRC_DIR = Path("/root/autodl-tmp/data/airbot_play_data")
DST_DIR = Path("/root/autodl-tmp/data/airbot_play_data_new")

# 读取原始数据集的 info
with open(SRC_DIR / "meta" / "info.json") as f:
    info = json.load(f)

fps = info["fps"]
num_episodes = info["total_episodes"]

# 加载原始数据集
src = LeRobotDataset("airbot_play_data", root=SRC_DIR)

# 创建新数据集
if DST_DIR.exists():
    shutil.rmtree(DST_DIR)

dst = LeRobotDataset.create(
    repo_id="airbot_play_data",
    robot_type="airbot_play",
    fps=fps,
    root=DST_DIR,
    features={
        "image": {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (480, 640, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
    },
    image_writer_threads=4,
    image_writer_processes=2,
)

# 逐 episode 复制数据
for ep_idx in range(num_episodes):
    ep_data = src.hf_dataset.filter(lambda x: x["episode_index"] == ep_idx)
    for i in range(len(ep_data)):
        row = ep_data[i]
        dst.add_frame({
            "image": np.array(row["image"]),
            "wrist_image": np.array(row["wrist_image"]),
            "state": np.array(row["state"], dtype=np.float32),
            "actions": np.array(row["actions"], dtype=np.float32),
            "task": src.meta.tasks[row["task_index"]],
        })
    dst.save_episode()
    print(f"Episode {ep_idx + 1}/{num_episodes} done")

print(f"新数据集已保存到: {DST_DIR}")

# 替换
shutil.rmtree(SRC_DIR)
DST_DIR.rename(SRC_DIR)
print("已替换原数据集")