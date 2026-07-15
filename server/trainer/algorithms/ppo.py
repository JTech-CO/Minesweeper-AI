"""Public PPO backend with strict bootstrap and training-seed boundaries."""

from __future__ import annotations

from pathlib import Path

from trainer.algorithms.ppo_core import PPOBackend as _PPOBackend
from trainer.config import ExperimentConfig
from trainer.evaluation.suites import training_seed
from trainer.storage.state import read_json


class PPOBackend(_PPOBackend):
    def __init__(self, config: ExperimentConfig, run_dir: Path) -> None:
        lifetime = read_json(Path(run_dir) / "lifetime.json", {"episodes": 0})
        self.seed_cursor = int(lifetime.get("episodes", 0))
        super().__init__(config, run_dir)

    def _restore_or_bootstrap(self) -> None:
        local_source = (self.run_dir / "last.pt").exists() or (
            self.run_dir / "bootstrap.pt"
        ).exists()
        production_arch = self.config.width == 128 and self.config.blocks == 8
        if local_source or production_arch:
            super()._restore_or_bootstrap()

    def _reset_env(self, index: int) -> None:
        seed = training_seed(self.seed_cursor, run_seed=self.config.seed)
        self.seed_cursor += 1
        env = self.envs[index]
        env.reset(seed)
        center = (env.rows // 2) * env.cols + env.cols // 2
        env.step(center)

