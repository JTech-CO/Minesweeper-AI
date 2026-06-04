"""Double DQN agent with n-step returns + a target network (기술백서 §4.3).

Action masking is applied to the Q-values *outside* the network and consistently in both
places it matters: ε-greedy action selection and the next-state argmax of the Double DQN
target. A finite mask value (not -inf) is used so that `masked_q * (1 - done)` can never
produce NaN for terminal transitions (run-book 3/4).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trainer.model import QNet

MASK_VALUE = -1.0e9


class NStepBuffer:
    """Accumulates the last n transitions into a single n-step transition."""

    def __init__(self, n_step: int, gamma: float) -> None:
        self.n = n_step
        self.gamma = gamma
        self.buf: deque = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buf.append((state, action, reward, next_state, done, next_mask))

    def pop_ready(self) -> list[tuple]:
        """Emit n-step transitions. On episode end, flush all remaining (shorter) ones."""
        out = []
        terminal = self.buf and self.buf[-1][4]
        while len(self.buf) == self.n or (terminal and self.buf):
            r, gamma_k = 0.0, 1.0
            for _, _, rew, _, d, _ in self.buf:
                r += gamma_k * rew
                gamma_k *= self.gamma
                if d:
                    break
            s0, a0 = self.buf[0][0], self.buf[0][1]
            # n-step landing state = last entry's next_state / done / mask
            last = self.buf[-1]
            out.append((s0, a0, r, last[3], last[4], last[5]))
            self.buf.popleft()
            if not terminal:
                break
            terminal = self.buf and self.buf[-1][4]
        return out

    def clear(self) -> None:
        self.buf.clear()


class DQNAgent:
    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        *,
        device: torch.device,
        width: int = 64,
        blocks: int = 4,
        lr: float = 5e-4,
        gamma: float = 0.99,
        n_step: int = 3,
        grad_clip: float = 10.0,
    ) -> None:
        self.device = device
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_step = n_step
        self.grad_clip = grad_clip
        self.online = QNet(in_channels, width, blocks).to(device)
        self.target = QNet(in_channels, width, blocks).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, state: np.ndarray, mask: np.ndarray, eps: float) -> int:
        """ε-greedy over LEGAL actions (mask: 1 = legal)."""
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            return 0  # no legal move (terminal); caller shouldn't act
        if np.random.random() < eps:
            return int(np.random.choice(legal))
        x = torch.from_numpy(state[None]).to(self.device)
        q = self.online(x).squeeze(0)
        mask_t = torch.from_numpy(mask.astype(np.bool_)).to(self.device)
        q = torch.where(mask_t, q, torch.full_like(q, MASK_VALUE))
        return int(torch.argmax(q).item())

    def learn(self, batch: dict, weights: np.ndarray) -> tuple[float, np.ndarray]:
        device = self.device
        states = torch.from_numpy(batch["states"]).to(device)
        actions = torch.from_numpy(batch["actions"]).to(device)
        rewards = torch.from_numpy(batch["rewards"]).to(device)
        next_states = torch.from_numpy(batch["next_states"]).to(device)
        dones = torch.from_numpy(batch["dones"]).to(device)
        next_masks = torch.from_numpy(batch["next_masks"]).to(device).bool()
        w = torch.from_numpy(weights).to(device)

        q = self.online(states).gather(1, actions[:, None]).squeeze(1)

        with torch.no_grad():
            # Double DQN: action chosen by online net, evaluated by target net — both masked.
            next_q_online = self.online(next_states)
            next_q_online = torch.where(
                next_masks, next_q_online, torch.full_like(next_q_online, MASK_VALUE)
            )
            next_actions = next_q_online.argmax(dim=1, keepdim=True)
            next_q_target = self.target(next_states).gather(1, next_actions).squeeze(1)
            not_done = (~dones).float()
            target = rewards + (self.gamma**self.n_step) * next_q_target * not_done

        td = target - q
        loss = (w * F.smooth_l1_loss(q, target, reduction="none")).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.opt.step()
        return float(loss.item()), td.detach().abs().cpu().numpy()

    def sync_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str, meta: dict | None = None) -> None:
        torch.save({"online": self.online.state_dict(), "meta": meta or {}}, path)

    def load(self, path: str) -> dict:
        # weights_only=False: we only load our own trusted checkpoints (torch 2.6 default).
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(ckpt["online"])
        self.sync_target()
        return ckpt.get("meta", {})
