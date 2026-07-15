from pathlib import Path

import torch

from trainer.algorithms.ppo import PPOBackend
from trainer.config import ExperimentConfig
from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.models import build_policy_model
from trainer.storage import save_checkpoint


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


def test_v5_ppo_uses_frozen_teacher_anchor_and_separate_mine_loss(
    tmp_path: Path,
) -> None:
    config = ExperimentConfig(
        algorithm="ppo",
        architecture="constraint-posterior-v5",
        graph_rounds=2,
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
        teacher_kl_coef=0.05,
        mine_aux_coef=0.1,
        action_temperature=0.25,
    )
    model_config = config.to_dict() | {
        "input_channels": NUM_CHANNELS_V3,
        "posterior_policy_weight": 0.5,
    }
    bootstrap_model = build_policy_model(model_config)
    manifest = save_checkpoint(
        tmp_path / "bootstrap.pt",
        model=bootstrap_model,
        config=model_config,
        run_id="v5-bootstrap",
        git_sha="test",
        episode=0,
        global_step=0,
    )
    backend = PPOBackend(config, tmp_path)
    assert backend.teacher_anchor is not None
    assert backend.anchor_checkpoint_id == manifest.checkpoint_id
    probabilities = backend._policy_distribution(
        torch.tensor([[0.0, 1.0]])
    ).probs
    assert probabilities[0, 1] > 0.95
    assert all(
        not parameter.requires_grad
        for parameter in backend.teacher_anchor.parameters()
    )
    anchor_before = {
        name: value.detach().clone()
        for name, value in backend.teacher_anchor.state_dict().items()
    }

    metrics = backend.train_update()

    assert metrics["teacher_kl"] >= 0
    assert torch.isfinite(torch.tensor(metrics["teacher_kl"]))
    assert torch.isfinite(torch.tensor(metrics["mine_loss"]))
    assert torch.isfinite(torch.tensor(metrics["policy_loss"]))
    assert all(
        torch.equal(anchor_before[name], value)
        for name, value in backend.teacher_anchor.state_dict().items()
    )
    backend.close()

    restored = PPOBackend(config, tmp_path)
    assert restored.anchor_checkpoint_id == manifest.checkpoint_id
    assert restored.action_temperature == 0.25
    assert restored.teacher_anchor is not None
    assert all(
        torch.equal(anchor_before[name], value)
        for name, value in restored.teacher_anchor.state_dict().items()
    )
    restored.close()