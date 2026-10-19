import numpy as np

from openpi.models import model as _model
from openpi.policies.hrc_policy import HrcR1ProHead
from openpi.policies.hrc_policy import HrcR1ProOutputs
from openpi.policies.hrc_policy import SelectHrcPrompt


def test_hrc_head_uses_only_head_camera_and_main_state():
    transform = HrcR1ProHead(model_type=_model.ModelType.PI05, state_dim=23)
    data = {
        "observation/image_0": np.full((3, 8, 10), 0.5, dtype=np.float32),
        "observation/state": np.arange(23, dtype=np.float32),
        "actions": np.ones((32, 23), dtype=np.float32),
        "prompt": "Task: Pick up the cup. Phase: PICKUP.",
    }

    output = transform(data)

    assert output["image"]["base_0_rgb"].shape == (8, 10, 3)
    assert output["image"]["base_0_rgb"].dtype == np.uint8
    assert tuple(bool(output["image_mask"][key]) for key in output["image_mask"]) == (True, False, False)
    np.testing.assert_array_equal(output["state"], data["observation/state"])
    np.testing.assert_array_equal(output["actions"], data["actions"])
    assert output["prompt"] == data["prompt"]


def test_hrc_head_accepts_privileged_state_when_configured():
    transform = HrcR1ProHead(model_type=_model.ModelType.PI05, state_dim=28)
    output = transform(
        {
            "observation/image_0": np.zeros((8, 10, 3), dtype=np.uint8),
            "observation/state": np.zeros(28, dtype=np.float32),
        }
    )
    assert output["state"].shape == (28,)


def test_hrc_head_rejects_wrong_state_dimension():
    transform = HrcR1ProHead(model_type=_model.ModelType.PI05, state_dim=23)
    with np.testing.assert_raises_regex(ValueError, "expected 23"):
        transform(
            {
                "observation/image_0": np.zeros((8, 10, 3), dtype=np.uint8),
                "observation/state": np.zeros(28, dtype=np.float32),
            }
        )


def test_hrc_outputs_slice_model_padding_to_r1pro_action():
    actions = np.arange(64 * 32, dtype=np.float32).reshape(2, 32, 32)
    output = HrcR1ProOutputs()({"actions": actions})
    assert output["actions"].shape == (2, 32, 23)
    np.testing.assert_array_equal(output["actions"], actions[..., :23])


def test_select_hrc_prompt_can_remove_phase_for_ablation():
    data = {"prompt": "Task: Follow the person. Phase: ExecuteBehaviorTask."}
    assert SelectHrcPrompt(include_phase=True)(data)["prompt"] == data["prompt"]
    assert SelectHrcPrompt(include_phase=False)(data)["prompt"] == "Follow the person."


def test_select_hrc_prompt_rejects_unexpected_phase_format():
    with np.testing.assert_raises_regex(ValueError, "phase-conditioned"):
        SelectHrcPrompt(include_phase=False)({"prompt": "Follow the person."})
