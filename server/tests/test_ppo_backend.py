from pathlib import Path

from trainer.algorithms.ppo import PPOBackend
from trainer.config import ExperimentConfig


def test_masked_ppo_update_writes_durable_metrics(tmp_path: Path) -> None:
    config = ExperimentConfig(
        algorithm="ppo",
        difficulty="beginner",
        device="cpu",
        learning_rate=1e-4,
        width=16,
        blocks=1,
        num_envs=2,
        rollout_steps=2,
        batch_size=4,
        total_updates=1,
        eval_every=99,
        checkpoint_every=99,
    )
    backend = PPOBackend(config, tmp_path)
    metrics = backend.train_update()
    assert metrics["steps"] == 4
    assert metrics["loss"] == metrics["loss"]
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "lifetime.json").exists()
    backend.close()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "last.pt.manifest.json").exists()

def test_expert_curriculum_rollout_rehearses_previous_difficulties(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        algorithm="ppo",
        difficulty="beginner",
        device="cpu",
        width=16,
        blocks=1,
        num_envs=8,
        rollout_steps=1,
        batch_size=8,
        total_updates=1,
        eval_every=99,
        checkpoint_every=99,
        curriculum=True,
    )
    backend = PPOBackend(config, tmp_path)
    assert backend.curriculum is not None
    backend.curriculum.stage_index = 2
    backend.current_difficulty = "expert"
    backend._configure_envs()

    assert backend.env_difficulties.count("expert") == 6
    assert set(backend.env_difficulties) == {
        "beginner",
        "intermediate",
        "expert",
    }
    states, legal = backend._batch()
    assert states.shape == (8, 20, 16, 30)
    assert legal.shape == (8, 16, 30)
    beginner_index = backend.env_difficulties.index("beginner")
    assert not legal[beginner_index, 9:, :].any()
    assert not legal[beginner_index, :, 9:].any()

    backend.outcomes.clear()
    backend._record_outcome(beginner_index, 1)
    assert not backend.outcomes
    backend._record_outcome(backend.env_difficulties.index("expert"), 1)
    assert list(backend.outcomes) == [1]
