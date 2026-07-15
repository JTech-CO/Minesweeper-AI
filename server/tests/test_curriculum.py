from __future__ import annotations

import pytest

from trainer.config import ExperimentConfig
from trainer.curriculum import CurriculumController


def _controller() -> CurriculumController:
    return CurriculumController(
        window=500,
        confirmations=3,
        beginner_gate=0.85,
        intermediate_gate=0.60,
        expert_target=0.40,
    )


def _observe(
    controller: CurriculumController,
    rate: float,
    *,
    games: int = 500,
    update: int = 1,
) -> dict | None:
    return controller.observe(
        win_rate=rate,
        games=games,
        global_update=update,
        episode=update * 500,
    )


def test_curriculum_requires_full_window_and_three_confirmations() -> None:
    controller = _controller()
    assert _observe(controller, 1.0, games=499) is None
    assert _observe(controller, 0.85) is None
    assert _observe(controller, 0.84) is None
    assert _observe(controller, 0.85) is None
    assert _observe(controller, 0.85) is None
    transition = _observe(controller, 0.85, update=6)

    assert transition is not None
    assert transition["from"] == "beginner"
    assert transition["to"] == "intermediate"
    assert controller.difficulty == "intermediate"
    assert controller.stage_updates == 0


def test_curriculum_state_roundtrip_and_expert_completion() -> None:
    controller = _controller()
    for _ in range(3):
        beginner_transition = _observe(controller, 0.90)
    assert beginner_transition is not None

    restored = _controller()
    restored.load_state_dict(controller.state_dict())
    assert restored.difficulty == "intermediate"
    assert restored.transitions == controller.transitions

    for _ in range(3):
        intermediate_transition = _observe(restored, 0.60)
    assert intermediate_transition is not None
    assert restored.difficulty == "expert"
    assert _observe(restored, 0.399) is None
    for _ in range(3):
        completion = _observe(restored, 0.40)
    assert completion is not None
    assert completion["type"] == "complete"
    assert completion["to"] is None
    assert restored.complete


def test_curriculum_config_preserves_documented_gates() -> None:
    config = ExperimentConfig(curriculum=True)
    assert config.curriculum_window == 500
    assert config.curriculum_confirmations == 3
    assert config.curriculum_beginner_gate == 0.85
    assert config.curriculum_intermediate_gate == 0.60
    assert config.curriculum_expert_target == 0.40
    with pytest.raises(ValueError):
        ExperimentConfig(algorithm="counter", curriculum=True)
    with pytest.raises(ValueError):
        ExperimentConfig(difficulty="intermediate", curriculum=True)
