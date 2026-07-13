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

