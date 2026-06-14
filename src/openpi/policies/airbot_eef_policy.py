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
