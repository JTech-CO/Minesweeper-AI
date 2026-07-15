from __future__ import annotations

import json

import torch

from trainer.algorithms.ppo import PPOBackend
from trainer.config import ExperimentConfig


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        algorithm="ppo",
        difficulty="beginner",
        device="cpu",
        learning_rate=1e-4,
        width=16,
        blocks=1,
        num_envs=1,
        rollout_steps=1,
        batch_size=1,
        total_updates=10,
        eval_every=99,
        checkpoint_every=99,
        curriculum=True,
        curriculum_window=2,
        curriculum_confirmations=1,
        curriculum_beginner_gate=0.5,
        curriculum_intermediate_gate=0.5,
        curriculum_expert_target=0.5,
        curriculum_entropy_restart=0.02,
        curriculum_entropy_decay_updates=4,
    )


def _weights(backend: PPOBackend) -> dict[str, torch.Tensor]:
    return {name: value.detach().clone() for name, value in backend.model.state_dict().items()}


def _force_gate(backend: PPOBackend) -> dict:
    backend.outcomes.extend([1, 1])
    transition = backend._observe_curriculum(1.0)
    assert transition is not None
    return transition


def test_curriculum_promotion_preserves_weights_and_resume_state(tmp_path) -> None:
    backend = PPOBackend(_config(), tmp_path)
    before = _weights(backend)
    old_optimizer = backend.optimizer

    transition = _force_gate(backend)

    assert transition["from"] == "beginner"
    assert transition["to"] == "intermediate"
    assert backend.current_difficulty == "intermediate"
    assert (backend.rows, backend.cols, backend.mines) == (16, 16, 40)
    assert backend.optimizer is not old_optimizer
    assert backend.entropy_coef == 0.02
    assert not backend.outcomes
    assert all(
        torch.equal(before[name], value)
        for name, value in backend.model.state_dict().items()
    )

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events] == ["curriculum"]
    assert not (tmp_path / ".events" / "checkpoints").exists()

    restored = PPOBackend(_config(), tmp_path)
    assert restored.current_difficulty == "intermediate"
    assert restored.curriculum is not None
    assert restored.curriculum.stage_updates == 0
    assert all(
        torch.equal(before[name], value)
        for name, value in restored.model.state_dict().items()
    )

    transition = _force_gate(restored)
    assert transition["to"] == "expert"
    assert (restored.rows, restored.cols, restored.mines) == (16, 30, 99)
    completion = _force_gate(restored)
    assert completion["type"] == "complete"
    assert restored.curriculum.complete

    events = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    checkpoint_events = [event for event in events if event["type"] == "checkpoint"]
    assert len(checkpoint_events) == 1
    assert checkpoint_events[0]["role"] == "curriculum-complete"
    assert len(list((tmp_path / ".events" / "checkpoints").glob("*.pt"))) == 1
    restored.close()
