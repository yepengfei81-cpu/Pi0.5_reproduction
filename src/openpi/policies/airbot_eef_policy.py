"""AirBot Play EEF(任务空间) 策略变换 —— 用于 UMI + 遥操作联合训练。

与 airbot_play_policy.py(关节空间 7D) 的区别：
  - state/action = 任务空间 10D = pos(3) + rot6d(6) + gripper(1)（W' 相对系）。
  - 多一个 env_mask：UMI 样本无环境相机(置零)，靠它把 base_0 相机槽 mask 掉；
    遥操作样本环境相机有效。
结构刻意对齐 airbot_play_policy.AirbotPlayInputs/Outputs，便于对照。
"""
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_airbot_eef_example() -> dict:
    return {
        "observation/state": np.random.rand(10).astype(np.float32),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/env_mask": np.float32(1.0),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotEEFInputs(transforms.DataTransformFn):
    """EEF 任务空间输入: 10D state + 双相机(env 槽按 env_mask 决定是否有效)。"""
    model_type: _model.ModelType = _model.ModelType.PI0
    # 方案C: 夹爪几何点云 (P,3) (TCP 系)。None=不用 gripper token。
    # 单把爪: 所有样本注入同一份 gripper_pc(常量)。
    gripper_pc: np.ndarray | None = None
    # 多把爪共训: gripper_clouds (G,P,3) 查表, 按每帧 observation/gripper_id 选第几把爪。
    # 与 gripper_pc 互斥(给了 gripper_clouds 就用它)。部署单爪时 gripper_id 缺省=0。
    gripper_clouds: np.ndarray | None = None

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # env_mask: 1=环境相机有效(遥操作), 0=无效(UMI)。部署时缺省视为有效。
        env_valid = bool(
            np.asarray(data.get("observation/env_mask", 1.0)).reshape(-1)[0] > 0.5
        )

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.bool_(env_valid),
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 部署/zero-shot: obs 直接带 gripper_pc(当前夹爪甚至未见爪的描述符), 优先用
        if data.get("observation/gripper_pc") is not None:
            inputs["gripper_pc"] = np.asarray(data["observation/gripper_pc"], np.float32)
        elif self.gripper_clouds is not None:
            gid = int(np.asarray(data.get("observation/gripper_id", 0)).reshape(-1)[0])
            gid = min(max(gid, 0), len(self.gripper_clouds) - 1)
            inputs["gripper_pc"] = np.asarray(self.gripper_clouds[gid], np.float32)
        elif self.gripper_pc is not None:
            inputs["gripper_pc"] = np.asarray(self.gripper_pc, np.float32)

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotEEFOutputs(transforms.DataTransformFn):
    """返回前 10 个动作维 (pos3 + rot6d6 + gripper1)。"""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :10])}
