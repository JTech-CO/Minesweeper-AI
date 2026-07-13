import numpy as np

from trainer.data.hard_teacher import balance_teacher_samples, profile_sample
from trainer.data.teacher import generate_teacher_episode
from trainer.encoding_v3 import NUM_CHANNELS_V3


def test_balancer_keeps_hard_state_buckets_and_upgrades_encoding() -> None:
    samples = generate_teacher_episode(9, 9, 10, seed=101)
    assert samples
    selected = balance_teacher_samples(samples, max_states=min(24, len(samples)))
    assert selected
    assert all(sample.state.shape[0] == NUM_CHANNELS_V3 for sample in selected)
    profiles = [profile_sample(sample) for sample in samples]
    assert all(profile.components >= 0 for profile in profiles)
    assert all(0 <= profile.progress_bin <= 3 for profile in profiles)


def test_teacher_policy_targets_remain_normalized_after_balancing() -> None:
    samples = generate_teacher_episode(9, 9, 10, seed=303)
    selected = balance_teacher_samples(samples, max_states=len(samples))
    for sample in selected:
        assert np.isclose(sample.policy_target.sum(), 1.0)
        assert np.all(sample.policy_target[~sample.legal_mask.astype(bool)] == 0)

