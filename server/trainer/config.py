"""Single source of truth for M4-R1+ experiment configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Literal

from trainer.env import DIFFICULTIES
from trainer.storage import config_hash

Algorithm = Literal["ppo", "counter"]


@dataclass(frozen=True)
class ExperimentConfig:
    algorithm: Algorithm = "ppo"
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

    def __post_init__(self) -> None:
        if self.algorithm not in {"ppo", "counter"}:
            raise ValueError(f"unsupported algorithm {self.algorithm!r}")
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
