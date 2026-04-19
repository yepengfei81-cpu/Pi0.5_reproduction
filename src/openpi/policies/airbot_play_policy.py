import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_airbot_play_example() -> dict:
    """Creates a random input example for the AirBot Play policy."""
    return {
        "observation/state": np.random.rand(7).astype(np.float32),
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
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
class AirbotPlayInputs(transforms.DataTransformFn):
    """Converts AirBot Play inputs to the model's expected format.

    AirBot Play: 6-DOF arm + 1 gripper = 7D state/action.
    """
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
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
class AirbotPlayOutputs(transforms.DataTransformFn):
    """Returns first 7 action dims (6 joints + 1 gripper)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :7])}