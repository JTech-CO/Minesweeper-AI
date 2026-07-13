from __future__ import annotations

import numpy as np
import torch

from trainer.data.teacher import generate_teacher_dataset
from trainer.encoding_v2 import NUM_CHANNELS_V2, encode_v2
from trainer.env import MinesweeperEnv
from trainer.models import SpatialPolicyValueNet
from trainer.pretrain import TeacherPretrainer


def test_encoding_v2_context_and_frontier():
    env = MinesweeperEnv(9, 9, 10, seed=4)
    env.step(40)
    state = encode_v2(env)
    assert state.shape == (NUM_CHANNELS_V2, 9, 9)
    assert np.all(state[14] == 1)
    assert 0 < state[11, 0, 0] < 1
    assert 0 < state[12, 0, 0] <= 1
    assert state[13].sum() > 0


def test_model_v2_outputs_and_masks_all_board_sizes():
    model = SpatialPolicyValueNet(NUM_CHANNELS_V2, width=16, blocks=2)
    for rows, cols in ((9, 9), (16, 16), (16, 30)):
        state = torch.zeros(2, NUM_CHANNELS_V2, rows, cols)
        state[:, 14] = 1
        legal = torch.ones(2, rows, cols, dtype=torch.bool)
        legal[:, 0, 0] = False
        output = model(state, legal)
        assert output["policy_logits"].shape == (2, rows * cols)
        assert output["risk_logits"].shape == (2, rows * cols)
        assert output["certainty_logits"].shape == (2, 3, rows, cols)
        assert output["value"].shape == (2,)
        assert torch.all(output["policy_logits"][:, 0] < -1e8)


def test_teacher_labels_are_sound_and_pretraining_is_finite():
    dataset = generate_teacher_dataset(5, 5, 3, boards=8, seed_base=1234)
    for sample in dataset.samples:
        certain_safe = sample.certainty_target == 1
        certain_mine = sample.certainty_target == 2
        assert not np.any(sample.mine_target[certain_safe])
        assert np.all(sample.mine_target[certain_mine] == 1)
        assert np.isclose(sample.policy_target.sum(), 1.0)

    model = SpatialPolicyValueNet(NUM_CHANNELS_V2, width=16, blocks=2)
    trainer = TeacherPretrainer(model, device=torch.device("cpu"), learning_rate=1e-3)
    metrics = trainer.train_epoch(dataset, batch_size=8)
    assert all(np.isfinite(value) for value in metrics.values())
