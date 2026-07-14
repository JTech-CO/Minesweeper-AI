import numpy as np

from trainer.data import TeacherDataset, TeacherSample
from trainer.encoding_v2 import VALID_CHANNEL
from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.pretrain_graph import _pad_teacher_dataset


def test_mixed_difficulty_padding_preserves_valid_and_ignore_masks() -> None:
    rows, cols = 9, 9
    state = np.zeros((NUM_CHANNELS_V3, rows, cols), dtype=np.float32)
    state[VALID_CHANNEL] = 1
    sample = TeacherSample(
        state=state,
        legal_mask=np.ones((rows, cols), dtype=np.uint8),
        policy_target=np.full((rows, cols), 1 / (rows * cols), dtype=np.float32),
        risk_target=np.zeros((rows, cols), dtype=np.float32),
        mine_target=np.zeros((rows, cols), dtype=np.float32),
        certainty_target=np.zeros((rows, cols), dtype=np.int64),
        value_target=0.0,
    )

    padded = _pad_teacher_dataset(TeacherDataset([sample]), (16, 30))[0]

    assert padded["state"].shape == (NUM_CHANNELS_V3, 16, 30)
    assert padded["legal_mask"].shape == (16, 30)
    assert padded["state"][VALID_CHANNEL, :rows, :cols].all()
    assert not padded["state"][VALID_CHANNEL, rows:, :].any()
    assert not padded["state"][VALID_CHANNEL, :, cols:].any()
    assert padded["certainty_target"][:rows, :cols].eq(0).all()
    assert padded["certainty_target"][rows:, :].eq(-100).all()
    assert padded["certainty_target"][:, cols:].eq(-100).all()
    assert abs(padded["policy_target"].sum().item() - 1.0) < 1e-6
