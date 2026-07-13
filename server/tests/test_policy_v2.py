from __future__ import annotations

import numpy as np
import torch

from trainer.encoding_v2 import NUM_CHANNELS_V2
from trainer.models import SpatialPolicyValueNet
from trainer.policy_v2 import SpatialInferencePolicy


def test_multi_head_policy_never_selects_an_illegal_cell():
    model = SpatialPolicyValueNet(NUM_CHANNELS_V2, width=16, blocks=2)
    policy = SpatialInferencePolicy(model, torch.device("cpu"))
    state = np.zeros((NUM_CHANNELS_V2, 5, 5), dtype=np.float32)
    state[14] = 1.0
    legal = np.zeros((5, 5), dtype=np.uint8)
    legal[1, 3] = 1
    assert policy.act(state, legal) == 8
