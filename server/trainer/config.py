"""Single source of truth for M4-R1+ experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from trainer.env import DIFFICULTIES
from trainer.storage import config_hash

Algorithm = Literal["ppo", "counter"]
Architecture = Literal["axial-v3", "constraint-graph-v4"]


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: Algorithm = "ppo"
    architecture: Architecture = "axial-v3"
    graph_rounds: int = 0
    difficulty: str = "beginner"
    seed: int = 0
    device: str = "cuda"
    learning_rate: float = 3e-4
    width: int = 96
    blocks: int = 6
    num_envs: int = 32
    rollout_steps: int = 64
    batch_size: int = 512
    total_updates: int = 10_000
    eval_every: int = 50
    checkpoint_every: int = 50
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    curriculum: bool = False
    curriculum_window: int = 500
    curriculum_confirmations: int = 3
    curriculum_beginner_gate: float = 0.85
    curriculum_intermediate_gate: float = 0.60
    curriculum_expert_target: float = 0.40
    curriculum_entropy_restart: float = 0.01
    curriculum_entropy_decay_updates: int = 100

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "counter"}:
            raise ValueError(f"unsupported algorithm {self.algorithm!r}")
        if self.architecture not in {"axial-v3", "constraint-graph-v4"}:
            raise ValueError(f"unsupported architecture {self.architecture!r}")
        if self.graph_rounds < 0:
            raise ValueError("graph_rounds must be non-negative")
        if self.architecture == "constraint-graph-v4" and self.graph_rounds < 1:
            raise ValueError("constraint-graph-v4 requires graph_rounds >= 1")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {self.difficulty!r}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.width < 8 or self.blocks < 1:
            raise ValueError("model width/blocks are too small")
        for name in (
            "num_envs",
            "rollout_steps",
            "batch_size",
            "total_updates",
            "eval_every",
            "checkpoint_every",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.gae_lambda <= 1:
            raise ValueError("gae_lambda must be in [0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("clip_ratio must be in (0, 1)")
        if self.curriculum and self.algorithm != "ppo":
            raise ValueError("curriculum requires the PPO algorithm")
        if self.curriculum and self.difficulty != "beginner":
            raise ValueError("curriculum must start at beginner")
        if self.curriculum_window < 1 or self.curriculum_confirmations < 1:
            raise ValueError("curriculum window and confirmations must be positive")
        for name in (
            "curriculum_beginner_gate",
            "curriculum_intermediate_gate",
            "curriculum_expert_target",
            "curriculum_entropy_restart",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.curriculum_entropy_decay_updates < 1:
            raise ValueError("curriculum_entropy_decay_updates must be positive")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def hash(self) -> str:
        return config_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: dict) -> ExperimentConfig:
        return cls(**value)

    @classmethod
    def from_json(cls, raw: str) -> ExperimentConfig:
        return cls.from_dict(json.loads(raw))
