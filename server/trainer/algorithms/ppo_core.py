"""Vectorized legal-action-masked PPO for the R2-A axial policy."""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from trainer.config import ExperimentConfig
from trainer.encoding_v3 import NUM_CHANNELS_V3, encode_v3
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.evaluation.axial import TorchAxialPolicy, evaluate_axial_policy
from trainer.evaluation.suites import get_suite, training_seed
from trainer.models import build_policy_model
from trainer.storage import load_checkpoint, save_checkpoint
from trainer.storage.state import atomic_write_json, read_json


class PPOBackend:
    def __init__(self, config: ExperimentConfig, run_dir: Path) -> None:
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(config.device)
        self.model = build_policy_model(
            config.to_dict() | {"input_channels": NUM_CHANNELS_V3}
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-4,
        )
        self.entropy_coef = config.entropy_coef
        self.risk_coef = 0.1
        self.update_index = 0
        self.best_eval = 0.0
        self.run_id = f"ppo-{uuid.uuid4()}"
        self.outcomes: deque[int] = deque(maxlen=500)
        self.lifetime_path = self.run_dir / "lifetime.json"
        lifetime = read_json(self.lifetime_path, {"episodes": 0, "steps": 0})
        self.total_episodes = int(lifetime.get("episodes", 0))
        self.total_steps = int(lifetime.get("steps", 0))
        self._restore_or_bootstrap()

        rows, cols, mines = DIFFICULTIES[config.difficulty]
        self.rows = rows
        self.cols = cols
        self.mines = mines
        self.envs = [
            MinesweeperEnv(rows, cols, mines, first_click_policy="safe-area")
            for _ in range(config.num_envs)
        ]
        for index in range(config.num_envs):
            self._reset_env(index)

    @property
    def _checkpoint_config(self) -> dict:
        return self.config.to_dict() | {"input_channels": NUM_CHANNELS_V3}

    def _restore_or_bootstrap(self) -> None:
        last = self.run_dir / "last.pt"
        bootstrap = self.run_dir / "bootstrap.pt"
        if not bootstrap.exists():
            checkpoint_name = (
                "graph-pretrained.pt"
                if self.config.architecture == "constraint-graph-v4"
                else "axial-pretrained.pt"
            )
            bootstrap = (
                Path(__file__).resolve().parents[2]
                / "storage"
                / "checkpoints"
                / "m4-r2"
                / checkpoint_name
            )
        source = last if last.exists() else bootstrap
        if not source.exists():
            return
        payload, _ = load_checkpoint(
            source,
            model=self.model,
            optimizer=self.optimizer if source == last else None,
            map_location=self.device,
        )
        if source == last:
            self.update_index = int(payload.get("extra_state", {}).get("update", 0))
            self.best_eval = float(payload.get("extra_state", {}).get("best_eval", 0.0))
            self.run_id = str(payload.get("run_id") or self.run_id)

    def _reset_env(self, index: int) -> None:
        seed = training_seed(self.total_episodes + index, run_seed=self.config.seed)
        env = self.envs[index]
        env.reset(seed)
        center = (self.rows // 2) * self.cols + self.cols // 2
        env.step(center)

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        states = np.stack([encode_v3(env) for env in self.envs])
        legal = np.stack(
            [env.legal_action_mask().reshape(self.rows, self.cols) for env in self.envs]
        )
        return (
            torch.from_numpy(states).to(self.device),
            torch.from_numpy(legal).to(self.device).bool(),
        )

    def _reward(self, before: float, after: float, status: str) -> float:
        shaping = 0.2 * (self.config.gamma * after - before)
        if status == "won":
            return 1.0 + shaping
        if status == "lost":
            return -1.0 + shaping
        return shaping - 0.002

    def _collect_rollout(self) -> dict[str, torch.Tensor | int]:
        states = []
        legal_masks = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        dones = []
        mine_targets = []
        completed = 0
        steps = 0

        self.model.eval()
        for _ in range(self.config.rollout_steps):
            state, legal = self._batch()
            with torch.no_grad():
                output = self.model(state, legal)
                distribution = Categorical(logits=output["policy_logits"])
                action = distribution.sample()
            mine_target = np.stack(
                [env.mine_layout.reshape(self.rows, self.cols) for env in self.envs]
            )
            step_rewards = []
            step_dones = []
            for index, env in enumerate(self.envs):
                before = env.safe_revealed / max(env.safe_cells, 1)
                _, _, done, _ = env.step(int(action[index].item()))
                after = env.safe_revealed / max(env.safe_cells, 1)
                step_rewards.append(self._reward(before, after, env.status))
                step_dones.append(float(done))
                steps += 1
                if done:
                    won = int(env.status == "won")
                    self.outcomes.append(won)
                    completed += 1
                    self.total_episodes += 1
                    self._reset_env(index)

            states.append(state.cpu())
            legal_masks.append(legal.cpu())
            actions.append(action.cpu())
            log_probs.append(distribution.log_prob(action).cpu())
            values.append(output["value"].cpu())
            rewards.append(torch.tensor(step_rewards, dtype=torch.float32))
            dones.append(torch.tensor(step_dones, dtype=torch.float32))
            mine_targets.append(torch.from_numpy(mine_target.astype(np.float32)))

        with torch.no_grad():
            next_state, next_legal = self._batch()
            next_value = self.model(next_state, next_legal)["value"].cpu()

        reward = torch.stack(rewards)
        done = torch.stack(dones)
        value = torch.stack(values)
        advantage = torch.zeros_like(reward)
        gae = torch.zeros(self.config.num_envs)
        for index in reversed(range(self.config.rollout_steps)):
            following = next_value if index == self.config.rollout_steps - 1 else value[index + 1]
            nonterminal = 1.0 - done[index]
            delta = reward[index] + self.config.gamma * following * nonterminal - value[index]
            gae = (
                delta
                + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            )
            advantage[index] = gae
        returns = advantage + value
        return {
            "state": torch.stack(states),
            "legal": torch.stack(legal_masks),
            "action": torch.stack(actions),
            "log_prob": torch.stack(log_probs),
            "advantage": advantage,
            "return": returns,
            "mine_target": torch.stack(mine_targets),
            "episodes": completed,
            "steps": steps,
        }

    def _optimize(self, rollout: dict[str, torch.Tensor | int]) -> float:
        count = self.config.rollout_steps * self.config.num_envs
        tensors = {}
        for name in (
            "state",
            "legal",
            "action",
            "log_prob",
            "advantage",
            "return",
            "mine_target",
        ):
            value = rollout[name]
            assert isinstance(value, torch.Tensor)
            tensors[name] = value.flatten(end_dim=1)
        advantage = tensors["advantage"]
        tensors["advantage"] = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        losses = []
        self.model.train()
        for _ in range(4):
            permutation = torch.randperm(count)
            for start in range(0, count, self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size]
                state = tensors["state"][indices].to(self.device)
                legal = tensors["legal"][indices].to(self.device)
                output = self.model(state, legal)
                distribution = Categorical(logits=output["policy_logits"])
                action = tensors["action"][indices].to(self.device)
                old_log_prob = tensors["log_prob"][indices].to(self.device)
                ratio = torch.exp(distribution.log_prob(action) - old_log_prob)
                batch_advantage = tensors["advantage"][indices].to(self.device)
                unclipped = ratio * batch_advantage
                clipped = torch.clamp(
                    ratio,
                    1 - self.config.clip_ratio,
                    1 + self.config.clip_ratio,
                ) * batch_advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = F.mse_loss(
                    output["value"], tensors["return"][indices].to(self.device)
                )
                legal_flat = legal.flatten(start_dim=1).bool()
                risk_loss = F.binary_cross_entropy_with_logits(
                    output["risk_logits"][legal_flat],
                    tensors["mine_target"][indices]
                    .to(self.device)
                    .flatten(start_dim=1)[legal_flat],
                )
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.risk_coef * risk_loss
                    - self.entropy_coef * entropy
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.detach().item()))
        return float(np.mean(losses))

    def _evaluate(self) -> dict:
        suite = get_suite("smoke")
        result = evaluate_axial_policy(
            TorchAxialPolicy(self.model, self.device),
            self.rows,
            self.cols,
            self.mines,
            seeds=suite.seeds,
            suite_name=suite.name,
        )
        return {
            "suite": result.suite,
            "games": result.games,
            "wins": result.wins,
            "win_rate": result.win_rate,
            "ci_low": result.ci_low,
            "ci_high": result.ci_high,
        }

    def _save(self, evaluation: dict | None = None) -> None:
        extra = {"update": self.update_index, "best_eval": self.best_eval}
        save_checkpoint(
            self.run_dir / "last.pt",
            model=self.model,
            optimizer=self.optimizer,
            config=self._checkpoint_config,
            run_id=self.run_id,
            git_sha="working-tree",
            episode=self.total_episodes,
            global_step=self.total_steps,
            current_eval=evaluation,
            best_eval={"win_rate": self.best_eval},
            extra_state=extra,
        )
        if evaluation and evaluation["win_rate"] >= self.best_eval:
            self.best_eval = float(evaluation["win_rate"])
            extra["best_eval"] = self.best_eval
            save_checkpoint(
                self.run_dir / "best.pt",
                model=self.model,
                optimizer=self.optimizer,
                config=self._checkpoint_config,
                run_id=self.run_id,
                git_sha="working-tree",
                episode=self.total_episodes,
                global_step=self.total_steps,
                current_eval=evaluation,
                best_eval=evaluation,
                extra_state=extra,
            )

    def _record_metrics(self, metrics: dict) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")

    def train_update(self) -> dict:
        rollout = self._collect_rollout()
        loss = self._optimize(rollout)
        self.update_index += 1
        steps = int(rollout["steps"])
        self.total_steps += steps
        atomic_write_json(
            self.lifetime_path,
            {"episodes": self.total_episodes, "steps": self.total_steps},
        )
        evaluation = None
        if self.update_index % self.config.eval_every == 0:
            evaluation = self._evaluate()
        if self.update_index % self.config.checkpoint_every == 0 or evaluation:
            self._save(evaluation)
        train_win_rate = sum(self.outcomes) / len(self.outcomes) if self.outcomes else 0.0
        metrics = {
            "time": time.time(),
            "update": self.update_index,
            "episodes": self.total_episodes,
            "steps": self.total_steps,
            "train_win_rate": train_win_rate,
            "loss": loss,
            "eval": evaluation,
        }
        self._record_metrics(metrics)
        return {
            "episodes": int(rollout["episodes"]),
            "steps": steps,
            "train_win_rate": train_win_rate,
            "loss": loss,
        }

    def apply_live_config(self, values: dict) -> None:
        if "learning_rate" in values:
            learning_rate = float(values["learning_rate"])
            if learning_rate <= 0:
                raise ValueError("learning_rate must be positive")
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate
        if "entropy_coef" in values:
            self.entropy_coef = max(0.0, float(values["entropy_coef"]))

    def close(self) -> None:
        self._save()

