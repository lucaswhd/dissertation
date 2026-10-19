import dataclasses
import re

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

_PHASE_PROMPT_PATTERN = re.compile(r"^Task: (.+) Phase: [A-Za-z][A-Za-z0-9_]*\.$")


@dataclasses.dataclass(frozen=True)
class SelectHrcPrompt(transforms.DataTransformFn):
    """Keep the recorded phase prompt or recover its instruction-only ablation."""

    include_phase: bool = True

    def __call__(self, data: dict) -> dict:
        if self.include_phase or "prompt" not in data:
            return data
        match = _PHASE_PROMPT_PATTERN.fullmatch(str(data["prompt"]))
        if match is None:
            raise ValueError(f"expected a phase-conditioned HRC prompt, got {data['prompt']!r}")
        return {**data, "prompt": match.group(1)}


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"expected a 3D image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class HrcR1ProHead(transforms.DataTransformFn):
    """Adapt one HRC head image and an already-extracted state for pi0/pi0.5."""

    model_type: _model.ModelType = _model.ModelType.PI05
    state_dim: int = 23

    def __call__(self, data: dict) -> dict:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"unsupported HRC model type: {self.model_type}")

        state = np.asarray(data["observation/state"])
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"expected {self.state_dim}-dimensional HRC state, got shape {state.shape}")

        head = _parse_image(data["observation/image_0"])
        blank = np.zeros_like(head)
        output = {
            "state": state,
            "image": {
                "base_0_rgb": head,
                "left_wrist_0_rgb": blank,
                "right_wrist_0_rgb": blank,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            output["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class HrcR1ProOutputs(transforms.DataTransformFn):
    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}
