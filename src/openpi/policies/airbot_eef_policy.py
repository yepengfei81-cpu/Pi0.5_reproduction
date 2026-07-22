"""AirBot Play EEF(任务空间) 策略变换 —— 用于 UMI + 遥操作联合训练。

与 airbot_play_policy.py(关节空间 7D) 的区别：
  - state/action = 任务空间 10D = pos(3) + rot6d(6) + gripper(1)（W' 相对系）。
  - 多一个 env_mask：UMI 样本无环境相机(置零)，靠它把 base_0 相机槽 mask 掉；
    遥操作样本环境相机有效。
结构刻意对齐 airbot_play_policy.AirbotPlayInputs/Outputs，便于对照。
"""
import dataclasses
import pathlib
import sys

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# 复用 gripper_geom/gripper_aug.py —— 训练与离线可视化共用同一套点云增强(单一真源)
_GG = pathlib.Path(__file__).resolve().parents[3] / "gripper_geom"
if str(_GG) not in sys.path:
    sys.path.insert(0, str(_GG))
try:
    from gripper_aug import augment_cloud as _augment_cloud
except Exception:  # noqa: BLE001
    _augment_cloud = None


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
    # 训练时是否对(查表/常量)夹爪点云做几何增强; 部署(obs 直接给点云)分支永不增强。
    augment: bool = False
    # 双臂 20D: state=20D, 第3相机(right_wrist=wrist_image_1, mask=arm1_mask), 夹爪 token 用臂0
    # (gripper_id_0)。臂1 的第2个 token 留到 pi0.py(stage2); 单臂样本 arm1_mask=0 屏蔽臂1相机。
    dual: bool = False

    def _aug(self, cloud: np.ndarray) -> np.ndarray:
        """train-only 几何增强 + 重采样回原点数(dropout 改了点数也保持固定 P)。"""
        if not self.augment:
            return cloud
        if _augment_cloud is None:
            raise RuntimeError("augment=True 但导入不到 gripper_geom/gripper_aug.py")
        P = cloud.shape[0]
        a = _augment_cloud(cloud)                     # finger_id=None -> 按 sign(Y) 开合; 用锁定默认范围
        if len(a) != P:
            idx = np.random.default_rng().choice(len(a), P, replace=len(a) < P)
            a = a[idx]
        return np.asarray(a, np.float32)

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # env_mask: 1=环境相机有效(遥操作), 0=无效(UMI)。部署时缺省视为有效。
        env_valid = bool(
            np.asarray(data.get("observation/env_mask", 1.0)).reshape(-1)[0] > 0.5
        )

        if self.dual:
            # 双臂: 第3相机 = 臂1 手眼(wrist_image_1), 其 mask = arm1_mask(单臂样本=0 屏蔽)。
            wrist1 = _parse_image(data["observation/wrist_image_1"]) if data.get(
                "observation/wrist_image_1") is not None else np.zeros_like(base_image)
            arm1_valid = bool(np.asarray(data.get("observation/arm1_mask", 1.0)).reshape(-1)[0] > 0.5)
            right_wrist, right_mask = wrist1, np.bool_(arm1_valid)
        else:
            right_wrist = np.zeros_like(base_image)
            right_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,     # 臂0 手眼
                "right_wrist_0_rgb": right_wrist,    # 双臂=臂1手眼 / 单臂=空图
            },
            "image_mask": {
                "base_0_rgb": np.bool_(env_valid),
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": right_mask,
            },
        }

        # 夹爪几何 token。臂0(单臂=唯一)走 gripper_id / gripper_id_0; 双臂再加臂1(gripper_id_1)。
        gid0_key = "observation/gripper_id_0" if self.dual else "observation/gripper_id"

        def _cloud_from(gid_key):
            gid = int(np.asarray(data.get(gid_key, 0)).reshape(-1)[0])
            gid = min(max(gid, 0), len(self.gripper_clouds) - 1)
            return self._aug(np.asarray(self.gripper_clouds[gid], np.float32))

        # 臂0: 部署/zero-shot obs 直接给点云优先(不增强), 否则查表(train 增强)
        if data.get("observation/gripper_pc") is not None:
            inputs["gripper_pc"] = np.asarray(data["observation/gripper_pc"], np.float32)
        elif self.gripper_clouds is not None:
            inputs["gripper_pc"] = _cloud_from(gid0_key)
        elif self.gripper_pc is not None:
            inputs["gripper_pc"] = self._aug(np.asarray(self.gripper_pc, np.float32))

        # 双臂: 臂1 token(gripper_id_1 或 obs 直接给) + arm1_mask(供模型屏蔽单臂样本的臂1 token)
        if self.dual:
            if data.get("observation/gripper_pc_1") is not None:
                inputs["gripper_pc_1"] = np.asarray(data["observation/gripper_pc_1"], np.float32)
            elif self.gripper_clouds is not None:
                inputs["gripper_pc_1"] = _cloud_from("observation/gripper_id_1")
            inputs["arm1_mask"] = np.asarray(
                data.get("observation/arm1_mask", 1.0), np.float32).reshape(1)

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotEEFOutputs(transforms.DataTransformFn):
    """返回动作维: 单臂前10维(pos3+rot6d6+grip1); 双臂前20维(臂0 10 + 臂1 10)。

    dual=False 时只吐臂0的10维(与单臂部署一致); dual=True 时吐 20 维,
    双臂部署据此把前10维发臂0、后10维发臂1。只影响推理后处理, 不改训练权重。
    """
    dual: bool = False

    def __call__(self, data: dict) -> dict:
        n = 20 if self.dual else 10
        return {"actions": np.asarray(data["actions"][:, :n])}
