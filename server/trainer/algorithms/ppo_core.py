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
from trainer.curriculum import CurriculumController
from trainer.encoding_v3 import NUM_CHANNELS_V3, encode_v3
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.evaluation.axial import TorchAxialPolicy, evaluate_axial_policy
from trainer.evaluation.suites import get_suite, training_seed
from trainer.events import append_event, emit_checkpoint, emit_metric
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
        self.optimizer = self._new_optimizer()
        self.curriculum = (
            CurriculumController(
                window=config.curriculum_window,
                confirmations=config.curriculum_confirmations,
                beginner_gate=config.curriculum_beginner_gate,
                intermediate_gate=config.curriculum_intermediate_gate,
                expert_target=config.curriculum_expert_target,
            )
            if config.curriculum
            else None
        )
        self.current_difficulty = config.difficulty
        self.entropy_coef = config.entropy_coef
        self.risk_coef = 0.1
        self.update_index = 0
        self.best_eval = 0.0
        self.run_id = f"ppo-{uuid.uuid4()}"
        outcome_window = config.curriculum_window if config.curriculum else 500
        self.outcomes: deque[int] = deque(maxlen=outcome_window)
        self.lifetime_path = self.run_dir / "lifetime.json"
        lifetime = read_json(self.lifetime_path, {"episodes": 0, "steps": 0})
        self.total_episodes = int(lifetime.get("episodes", 0))
        self.total_steps = int(lifetime.get("steps", 0))
        self._restore_or_bootstrap()
        self._configure_envs()

    def _new_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4,
        )

    def _configure_envs(self) -> None:
        self.rows, self.cols, self.mines = DIFFICULTIES[self.current_difficulty]
        stages = list(DIFFICULTIES)
        previous = stages[: stages.index(self.current_difficulty)]
        rehearsal_count = 0
        if (
            self.curriculum is not None
            and previous
            and self.config.curriculum_rehearsal_fraction > 0
        ):
            rehearsal_count = min(
                self.config.num_envs - 1,
                max(
                    1,
                    round(
                        self.config.num_envs
                        * self.config.curriculum_rehearsal_fraction
                    ),
                ),
            )
        self.env_difficulties = [self.current_difficulty] * (
            self.config.num_envs - rehearsal_count
        )
        self.env_difficulties.extend(
            previous[index % len(previous)] for index in range(rehearsal_count)
        )
        self.envs = [
            MinesweeperEnv(
                *DIFFICULTIES[difficulty],
                first_click_policy="safe-area",
            )
            for difficulty in self.env_difficulties
        ]
        for index in range(self.config.num_envs):
            self._reset_env(index)

    @property
    def _checkpoint_config(self) -> dict:
        return self.config.to_dict() | {
            "input_channels": NUM_CHANNELS_V3,
            "current_difficulty": self.current_difficulty,
        }

    def _extra_state(self) -> dict:
        extra = {"update": self.update_index, "best_eval": self.best_eval}
        if self.curriculum is not None:
            extra.update(
                {
                    "curriculum": self.curriculum.state_dict(),
                    "current_difficulty": self.current_difficulty,
                    "recent_outcomes": list(self.outcomes),
                    "entropy_coef": self.entropy_coef,
                }
            )
        return extra

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
            extra = payload.get("extra_state", {})
            self.update_index = int(extra.get("update", 0))
            self.best_eval = float(extra.get("best_eval", 0.0))
            self.run_id = str(payload.get("run_id") or self.run_id)
            if self.curriculum is not None and extra.get("curriculum"):
                self.curriculum.load_state_dict(extra["curriculum"])
                self.current_difficulty = self.curriculum.difficulty
                self.outcomes.extend(
                    int(value) for value in extra.get("recent_outcomes", [])
                )
                self.entropy_coef = float(
                    extra.get("entropy_coef", self.config.entropy_coef)
                )

    def _reset_env(self, index: int) -> None:
        seed = training_seed(self.total_episodes + index, run_seed=self.config.seed)
        env = self.envs[index]
        env.reset(seed)
        center = (env.rows // 2) * env.cols + env.cols // 2
        env.step(center)

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        states = np.stack(
            [encode_v3(env, (self.rows, self.cols)) for env in self.envs]
        )
        legal = np.zeros(
            (self.config.num_envs, self.rows, self.cols), dtype=np.uint8
        )
        for index, env in enumerate(self.envs):
            legal[index, : env.rows, : env.cols] = env.legal_action_mask().reshape(
                env.rows, env.cols
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

    def _record_outcome(self, index: int, won: int) -> None:
        if self.env_difficulties[index] == self.current_difficulty:
            self.outcomes.append(won)

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
            mine_target = np.zeros(
                (self.config.num_envs, self.rows, self.cols), dtype=np.uint8
            )
            for index, env in enumerate(self.envs):
                mine_target[index, : env.rows, : env.cols] = env.mine_layout.reshape(
                    env.rows, env.cols
                )
            step_rewards = []
            step_dones = []
            for index, env in enumerate(self.envs):
                before = env.safe_revealed / max(env.safe_cells, 1)
                target_action = int(action[index].item())
                action_row, action_col = divmod(target_action, self.cols)
                local_action = action_row * env.cols + action_col
                _, _, done, _ = env.step(local_action)
                after = env.safe_revealed / max(env.safe_cells, 1)
                step_rewards.append(self._reward(before, after, env.status))
                step_dones.append(float(done))
                steps += 1
                if done:
                    won = int(env.status == "won")
                    self._record_outcome(index, won)
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

    def _save(self, evaluation: dict | None = None, *, publish: bool = False) -> None:
        extra = self._extra_state()
        last_path = self.run_dir / "last.pt"
        last_manifest = save_checkpoint(
            last_path,
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
        if publish or self.curriculum is None:
            role = "curriculum-complete" if publish and self.curriculum else "last"
            emit_checkpoint(
                self.run_dir,
                source=last_path,
                manifest=last_manifest,
                role=role,
            )
        if evaluation and evaluation["win_rate"] >= self.best_eval:
            self.best_eval = float(evaluation["win_rate"])
            extra["best_eval"] = self.best_eval
            best_path = self.run_dir / "best.pt"
            best_manifest = save_checkpoint(
                best_path,
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
            if publish or self.curriculum is None:
                emit_checkpoint(
                    self.run_dir,
                    source=best_path,
                    manifest=best_manifest,
                    role="best",
                )

    def _update_curriculum_entropy(self) -> None:
        if self.curriculum is None or not self.curriculum.transitions:
            return
        fraction = min(
            1.0,
            self.curriculum.stage_updates
            / self.config.curriculum_entropy_decay_updates,
        )
        restart = self.config.curriculum_entropy_restart
        self.entropy_coef = restart + (self.config.entropy_coef - restart) * fraction

    def _record_transition(self, transition: dict) -> None:
        record = {
            "time": time.time(),
            "run_id": self.run_id,
            **transition,
        }
        append_event(self.run_dir / "curriculum.jsonl", record)
        append_event(
            self.run_dir / "events.jsonl",
            {
                "version": 1,
                "type": "curriculum",
                "run_id": self.run_id,
                "time": record["time"],
                "payload": transition,
            },
        )

    def _promote(self) -> None:
        assert self.curriculum is not None
        self.current_difficulty = self.curriculum.difficulty
        self.best_eval = 0.0
        self.outcomes.clear()
        self.optimizer = self._new_optimizer()
        self.entropy_coef = self.config.curriculum_entropy_restart
        self._configure_envs()

    def _observe_curriculum(self, train_win_rate: float) -> dict | None:
        if self.curriculum is None:
            return None
        transition = self.curriculum.observe(
            win_rate=train_win_rate,
            games=len(self.outcomes),
            global_update=self.update_index,
            episode=self.total_episodes,
        )
        if transition is None:
            return None
        self._record_transition(transition)
        if transition["type"] == "promotion":
            self._promote()
        self._save(publish=transition["type"] == "complete")
        return transition

    def _record_metrics(self, metrics: dict) -> None:
        with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        emit_metric(
            self.run_dir,
            run_id=self.run_id,
            config=self._checkpoint_config,
            metrics=metrics,
        )

    def train_update(self) -> dict:
        self._update_curriculum_entropy()
        stage_difficulty = self.current_difficulty
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
        transition = self._observe_curriculum(train_win_rate)
        complete = bool(self.curriculum and self.curriculum.complete)
        metrics = {
            "time": time.time(),
            "update": self.update_index,
            "episodes": self.total_episodes,
            "steps": self.total_steps,
            "difficulty": stage_difficulty,
            "current_difficulty": self.current_difficulty,
            "stage_update": (
                None if self.curriculum is None else self.curriculum.stage_updates
            ),
            "curriculum_complete": complete,
            "transition": transition,
            "entropy_coef": self.entropy_coef,
            "rehearsal_envs": sum(
                difficulty != self.current_difficulty
                for difficulty in self.env_difficulties
            ),
            "train_win_rate": train_win_rate,
            "loss": loss,
            "eval": evaluation,
        }
        self._record_metrics(metrics)
        return {
            "episodes": int(rollout["episodes"]),
            "steps": steps,
            "difficulty": self.current_difficulty,
            "stage_update": metrics["stage_update"],
            "curriculum_complete": complete,
            "transition": transition,
            "train_win_rate": train_win_rate,
            "loss": loss,
            "complete": complete,
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

