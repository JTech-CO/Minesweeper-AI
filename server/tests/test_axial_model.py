import numpy as np
import torch

from trainer.encoding_v2 import VALID_CHANNEL
from trainer.encoding_v3 import NUM_CHANNELS_V3, encode_v3
from trainer.env import MinesweeperEnv
from trainer.models.policy_value_axial import AxialPolicyValueNet


def test_encoding_v3_adds_visible_constraint_planes() -> None:
    env = MinesweeperEnv(9, 9, 10, seed=41, first_click_policy="safe-area")
    env.reset()
    env.step(40)
    encoded = encode_v3(env)
    assert encoded.shape == (NUM_CHANNELS_V3, 9, 9)
    assert np.array_equal(encoded[VALID_CHANNEL], np.ones((9, 9), dtype=np.float32))
    assert encoded[15].sum() > 0
    assert encoded[16:20].min() >= 0
    assert encoded[16:20].max() <= 1


def test_axial_model_masks_illegal_actions_for_all_board_sizes() -> None:
    model = AxialPolicyValueNet(NUM_CHANNELS_V3, width=32, blocks=2)
    for rows, cols in ((9, 9), (16, 30)):
        state = torch.zeros(2, NUM_CHANNELS_V3, rows, cols)
        state[:, VALID_CHANNEL] = 1
        legal = torch.zeros(2, rows, cols, dtype=torch.bool)
        legal[:, 1, 2] = True
        output = model(state, legal)
        assert output["policy_logits"].shape == (2, rows * cols)
        assert output["risk_logits"].shape == (2, rows * cols)
        assert output["certainty_logits"].shape == (2, 3, rows, cols)
        assert output["value"].shape == (2,)
        assert torch.all(output["policy_logits"][:, 1 * cols + 2] > -1e8)
        assert torch.all(output["policy_logits"][:, 0] < -1e8)

