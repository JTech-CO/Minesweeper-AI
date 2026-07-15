"""Deterministic Beginner -> Intermediate -> Expert curriculum state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CurriculumStage:
    difficulty: str
    threshold: float


class CurriculumController:
    def __init__(
        self,
        *,
        window: int,
        confirmations: int,
        beginner_gate: float,
        intermediate_gate: float,
        expert_target: float,
    ) -> None:
        if window < 1 or confirmations < 1:
            raise ValueError("curriculum window and confirmations must be positive")
        for value in (beginner_gate, intermediate_gate, expert_target):
            if not 0 <= value <= 1:
                raise ValueError("curriculum thresholds must be in [0, 1]")
        self.window = window
        self.confirmations_required = confirmations
        self.stages = (
            CurriculumStage("beginner", beginner_gate),
            CurriculumStage("intermediate", intermediate_gate),
            CurriculumStage("expert", expert_target),
        )
        self.stage_index = 0
        self.stage_updates = 0
        self.confirmations = 0
        self.complete = False
        self.transitions: list[dict[str, Any]] = []

    @property
    def stage(self) -> CurriculumStage:
        return self.stages[self.stage_index]

    @property
    def difficulty(self) -> str:
        return self.stage.difficulty

    def observe(
        self,
        *,
        win_rate: float,
        games: int,
        global_update: int,
        episode: int,
    ) -> dict[str, Any] | None:
        if self.complete:
            return None
        if not 0 <= win_rate <= 1:
            raise ValueError("win_rate must be in [0, 1]")
        self.stage_updates += 1
        if games < self.window:
            self.confirmations = 0
            return None
        if win_rate >= self.stage.threshold:
            self.confirmations += 1
        else:
            self.confirmations = 0
            return None
        if self.confirmations < self.confirmations_required:
            return None

        previous = self.difficulty
        target = self.stage.threshold
        if self.stage_index == len(self.stages) - 1:
            self.complete = True
            event_type = "complete"
            following = None
        else:
            self.stage_index += 1
            self.stage_updates = 0
            self.confirmations = 0
            event_type = "promotion"
            following = self.difficulty
        event = {
            "type": event_type,
            "from": previous,
            "to": following,
            "win_rate": win_rate,
            "games": games,
            "threshold": target,
            "global_update": global_update,
            "episode": episode,
        }
        self.transitions.append(event)
        return event

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_index": self.stage_index,
            "stage_updates": self.stage_updates,
            "confirmations": self.confirmations,
            "complete": self.complete,
            "transitions": self.transitions,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        stage_index = int(state.get("stage_index", 0))
        if not 0 <= stage_index < len(self.stages):
            raise ValueError("invalid curriculum stage index")
        self.stage_index = stage_index
        self.stage_updates = max(0, int(state.get("stage_updates", 0)))
        self.confirmations = max(0, int(state.get("confirmations", 0)))
        self.complete = bool(state.get("complete", False))
        self.transitions = [dict(item) for item in state.get("transitions", [])]
