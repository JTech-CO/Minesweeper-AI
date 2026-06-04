"""In-process training manager — the control plane for the integrated dashboard.

Owns the DQN training loop in a background thread so the dashboard can pause/resume it,
change hyperparameters live (learning rate, batch, train-freq, ε floor), and read its
state. Training weights are snapshotted to a CPU "play net" so the dashboard's parallel
play sessions never contend with the GPU training. The cumulative number of trained
episodes (the model's lifetime session count) is persisted to disk and reloaded on start,
so it keeps counting across dashboard restarts (counted from when the model was created).

GPU stats are read-only (NVML via nvidia-ml-py); no overclocking is performed.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from trainer.dqn import MASK_VALUE, DQNAgent, NStepBuffer
from trainer.encoding import NUM_CHANNELS, encode
from trainer.env import DIFFICULTIES, MinesweeperEnv
from trainer.evaluate import evaluate
from trainer.model import QNet
from trainer.replay import PrioritizedReplay


@dataclass
class LiveConfig:
    """Hyperparameters that can be changed without restarting (weights preserved)."""

    lr: float = 5e-4
    batch: int = 256
    train_freq: int = 4
    eps_end: float = 0.05


class GpuMonitor:
    """Read-only NVML GPU stats. Degrades gracefully if NVML / a GPU is unavailable."""

    def __init__(self) -> None:
        self._h = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self._h)
        except Exception:  # noqa: BLE001 - monitoring is best-effort
            self._nvml = None
            self.name = "GPU"

    def stats(self) -> dict | None:
        if self._h is None:
            return None
        try:
            n = self._nvml
            mem = n.nvmlDeviceGetMemoryInfo(self._h)
            util = n.nvmlDeviceGetUtilizationRates(self._h)
            return {
                "name": self.name,
                "vram_used_mb": mem.used // 1048576,
                "vram_total_mb": mem.total // 1048576,
                "vram_pct": round(100 * mem.used / mem.total, 1),
                "util_pct": util.gpu,
                "temp_c": n.nvmlDeviceGetTemperature(self._h, n.NVML_TEMPERATURE_GPU),
                "clock_mhz": n.nvmlDeviceGetClockInfo(self._h, n.NVML_CLOCK_GRAPHICS),
            }
        except Exception:  # noqa: BLE001
            return None


class TrainingManager:
    EPS_START = 1.0
    EPS_DECAY_STEPS = 150_000
    GAMMA = 0.99
    N_STEP = 3
    CAPACITY = 100_000
    WARMUP = 5_000
    TARGET_SYNC = 2_000
    EVAL_EVERY = 2_000
    EVAL_GAMES = 200
    PERSIST_EVERY = 2_000
    PLAY_SYNC_STEPS = 400

    def __init__(
        self,
        out_dir: str,
        *,
        tag: str = "managed",
        difficulty: str = "beginner",
        device: str = "cuda",
        width: int = 64,
        blocks: int = 4,
        bootstrap_ckpt: str | None = None,
        bootstrap_episode: int = 0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.tag = tag
        self.difficulty = difficulty
        self.rows, self.cols, self.mines = DIFFICULTIES[difficulty]
        self.n_actions = self.rows * self.cols
        self.width = width
        self.blocks = blocks
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.cfg = LiveConfig()
        self._lock = threading.Lock()
        self._play_lock = threading.Lock()
        self._pause = threading.Event()
        self._pause.set()  # set = running
        self._stop = threading.Event()
        self.status = "idle"

        self.cumulative_episodes = 0
        self.created_episode = 0
        self.global_step = 0
        self.best_eval = -1.0
        self.metrics: list[dict] = []
        self.latest: dict = {}
        self._t0 = time.time()
        self._steps_at_t0 = 0

        self.agent = DQNAgent(
            NUM_CHANNELS,
            self.n_actions,
            device=self.device,
            width=width,
            blocks=blocks,
            lr=self.cfg.lr,
            gamma=self.GAMMA,
            n_step=self.N_STEP,
        )
        self.replay = PrioritizedReplay(
            self.CAPACITY, (NUM_CHANNELS, self.rows, self.cols), self.n_actions
        )
        self.play_net = QNet(NUM_CHANNELS, width, blocks).to("cpu").eval()

        self.gpu = GpuMonitor()
        self._thread: threading.Thread | None = None
        self._bootstrap(bootstrap_ckpt, bootstrap_episode)
        self._sync_play_net()

    # -- persistence ----------------------------------------------------------

    @property
    def _state_path(self) -> Path:
        return self.out_dir / f"{self.tag}_state.json"

    @property
    def _last_path(self) -> Path:
        return self.out_dir / f"{self.tag}_last.pt"

    @property
    def _best_path(self) -> Path:
        return self.out_dir / f"{self.tag}_best.pt"

    def _bootstrap(self, bootstrap_ckpt: str | None, bootstrap_episode: int) -> None:
        if self._state_path.exists() and self._last_path.exists():
            try:
                state = json.loads(self._state_path.read_text())
                self.cumulative_episodes = state.get("cumulative_episodes", 0)
                self.created_episode = state.get("created_episode", 0)
                self.global_step = state.get("global_step", 0)
                self.best_eval = state.get("best_eval", -1.0)
                self.metrics = state.get("metrics", [])
                ckpt = torch.load(self._last_path, map_location=self.device, weights_only=False)
                self.agent.online.load_state_dict(ckpt["online"])
                self.agent.target.load_state_dict(ckpt.get("target", ckpt["online"]))
                if "optimizer" in ckpt:
                    self.agent.opt.load_state_dict(ckpt["optimizer"])
                return
            except Exception:  # noqa: BLE001 - fall through to bootstrap
                pass
        if bootstrap_ckpt and Path(bootstrap_ckpt).exists():
            try:
                ckpt = torch.load(bootstrap_ckpt, map_location=self.device, weights_only=False)
                self.agent.online.load_state_dict(ckpt["online"])
                self.agent.sync_target()
                meta = ckpt.get("meta", {})
                self.cumulative_episodes = bootstrap_episode or int(meta.get("episode", 0))
                self.created_episode = self.cumulative_episodes
                self.best_eval = float(meta.get("eval_wr", -1.0))
                # Pre-trained model → keep ε at its floor (don't re-explore from scratch).
                self.global_step = self.EPS_DECAY_STEPS
            except Exception:  # noqa: BLE001
                pass

    def _save(self, *, best: bool = False) -> None:
        payload = {
            "online": self.agent.online.state_dict(),
            "target": self.agent.target.state_dict(),
            "optimizer": self.agent.opt.state_dict(),
            "meta": {
                "episode": self.cumulative_episodes,
                "eval_wr": self.best_eval,
                "rows": self.rows,
                "cols": self.cols,
                "mines": self.mines,
                "in_channels": NUM_CHANNELS,
                "width": self.width,
                "blocks": self.blocks,
            },
        }
        torch.save(payload, self._last_path)
        if best:
            torch.save(payload, self._best_path)
        state = {
            "cumulative_episodes": self.cumulative_episodes,
            "created_episode": self.created_episode,
            "global_step": self.global_step,
            "best_eval": self.best_eval,
            "difficulty": self.difficulty,
            "width": self.width,
            "blocks": self.blocks,
            "metrics": self.metrics[-400:],
        }
        self._state_path.write_text(json.dumps(state))

    # -- control --------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._pause.set()
        self.status = "running"
        self._t0 = time.time()
        self._steps_at_t0 = self.global_step
        self._thread = threading.Thread(target=self._train_loop, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self._pause.clear()
        self.status = "paused"

    def resume(self) -> None:
        self._pause.set()
        self.status = "running"
        self._t0 = time.time()
        self._steps_at_t0 = self.global_step

    def stop(self) -> None:
        self._stop.set()
        self._pause.set()

    def set_config(self, **kw) -> None:
        with self._lock:
            if "lr" in kw and kw["lr"]:
                self.cfg.lr = float(kw["lr"])
                for g in self.agent.opt.param_groups:
                    g["lr"] = self.cfg.lr
            if "batch" in kw and kw["batch"]:
                self.cfg.batch = max(8, int(kw["batch"]))
            if "train_freq" in kw and kw["train_freq"]:
                self.cfg.train_freq = max(1, int(kw["train_freq"]))
            if "eps_end" in kw and kw["eps_end"] is not None:
                self.cfg.eps_end = max(0.0, min(1.0, float(kw["eps_end"])))

    # -- play net (CPU snapshot for the dashboard) ----------------------------

    def _sync_play_net(self) -> None:
        with self._play_lock:
            self.play_net.load_state_dict(
                {k: v.detach().cpu() for k, v in self.agent.online.state_dict().items()}
            )

    @torch.no_grad()
    def play_act(self, state: np.ndarray, mask: np.ndarray) -> int:
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0
        with self._play_lock:
            q = self.play_net(torch.from_numpy(state[None])).squeeze(0)
        mask_t = torch.from_numpy(mask.astype(np.bool_))
        q = torch.where(mask_t, q, torch.full_like(q, MASK_VALUE))
        return int(torch.argmax(q).item())

    # -- training loop --------------------------------------------------------

    def _eps(self) -> float:
        frac = min(1.0, self.global_step / self.EPS_DECAY_STEPS)
        return max(self.cfg.eps_end, self.EPS_START + (self.cfg.eps_end - self.EPS_START) * frac)

    def _train_loop(self) -> None:
        env = MinesweeperEnv(self.rows, self.cols, self.mines, seed=0)
        nstep = NStepBuffer(self.N_STEP, self.GAMMA)
        recent = []
        last_loss = float("nan")

        while not self._stop.is_set():
            self._pause.wait()  # blocks while paused
            if self._stop.is_set():
                break
            self.cumulative_episodes += 1
            env.reset(seed=self.cumulative_episodes)
            nstep.clear()
            state = encode(env)
            mask = env.legal_action_mask()
            done = False
            info = {"won": False}
            while not done:
                eps = self._eps()
                action = self.agent.act(state, mask, eps)
                _, reward, done, info = env.step(action)
                nxt = encode(env)
                nmask = env.legal_action_mask()
                nstep.push(state, action, reward, nxt, done, nmask)
                for tr in nstep.pop_ready():
                    self.replay.add(*tr)
                state, mask = nxt, nmask
                self.global_step += 1
                with self._lock:
                    batch_n, train_freq = self.cfg.batch, self.cfg.train_freq
                if len(self.replay) >= self.WARMUP and self.global_step % train_freq == 0:
                    batch = self.replay.sample(batch_n, beta=0.6)
                    last_loss, td = self.agent.learn(batch, batch["weights"])
                    self.replay.update_priorities(batch["indices"], td)
                    if self.global_step % self.TARGET_SYNC == 0:
                        self.agent.sync_target()
                if self.global_step % self.PLAY_SYNC_STEPS == 0:
                    self._sync_play_net()

            recent.append(1 if info["won"] else 0)
            recent = recent[-500:]
            sps = (self.global_step - self._steps_at_t0) / max(time.time() - self._t0, 1e-9)
            with self._lock:
                self.latest = {
                    "train_wr": float(np.mean(recent)) if recent else 0.0,
                    "loss": round(last_loss, 4),
                    "eps": round(self._eps(), 3),
                    "sps": round(sps),
                }

            if self.cumulative_episodes % self.EVAL_EVERY == 0:
                m = evaluate(
                    self.agent,
                    self.rows,
                    self.cols,
                    self.mines,
                    games=self.EVAL_GAMES,
                    seed_base=900_000_000,
                )
                eval_wr = m["win_rate"]
                self._sync_play_net()
                with self._lock:
                    self.metrics.append(
                        {
                            "episode": self.cumulative_episodes,
                            "eval_wr": eval_wr,
                            "train_wr": self.latest.get("train_wr", 0.0),
                        }
                    )
                    self.metrics = self.metrics[-2000:]
                improved = eval_wr > self.best_eval
                if improved:
                    self.best_eval = eval_wr
                self._save(best=improved)
            elif self.cumulative_episodes % self.PERSIST_EVERY == 0:
                self._save()

        self._save()
        self.status = "stopped"

    # -- snapshot for the API -------------------------------------------------

    def snapshot(self) -> dict:
        with self._lock:
            latest = dict(self.latest)
            metrics = list(self.metrics)
            cfg = asdict(self.cfg)
        return {
            "status": self.status,
            "difficulty": self.difficulty,
            "cumulative_episodes": self.cumulative_episodes,
            "created_episode": self.created_episode,
            "trained_since_creation": self.cumulative_episodes - self.created_episode,
            "global_step": self.global_step,
            "best_eval": round(self.best_eval, 4) if self.best_eval >= 0 else None,
            "latest": latest,
            "metrics": metrics,
            "config": cfg,
            "gpu": self.gpu.stats(),
            "board": {"rows": self.rows, "cols": self.cols, "mines": self.mines},
        }
