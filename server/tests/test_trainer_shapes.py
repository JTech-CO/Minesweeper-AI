"""Fast wiring checks for the trainer (encoding / model / agent / replay / learn).

These catch shape and masking bugs without a long training run; the actual learning is
validated by a sanity-overfit run and the Beginner win-rate gate (M4 DoD #1)."""

from __future__ import annotations

import numpy as np
import torch

from trainer.dqn import DQNAgent
from trainer.encoding import NUM_CHANNELS, encode
from trainer.env import MinesweeperEnv
from trainer.model import QNet
from trainer.replay import PrioritizedReplay


def test_encode_shape_and_onehot():
    env = MinesweeperEnv(9, 9, 10, seed=1)
    env.step(40)
    s = encode(env)
    assert s.shape == (NUM_CHANNELS, 9, 9)
    assert s.dtype == np.float32
    # Channels 0..9 (hidden + 9 number one-hots) partition every cell: exactly one is 1.
    per_cell = s[:10].reshape(10, -1).sum(axis=0)
    assert np.all(per_cell == 1.0)


def test_model_forward_shape():
    net = QNet(NUM_CHANNELS, width=16, blocks=2)
    q = net(torch.zeros(4, NUM_CHANNELS, 9, 9))
    assert q.shape == (4, 81)


def test_agent_acts_only_on_legal_cells():
    agent = DQNAgent(NUM_CHANNELS, 81, device=torch.device("cpu"), width=16, blocks=2)
    env = MinesweeperEnv(9, 9, 10, seed=2)
    env.step(40)
    mask = env.legal_action_mask()
    for eps in (0.0, 1.0):
        action = agent.act(encode(env), mask, eps=eps)
        assert mask[action] == 1


def test_learn_step_is_finite():
    agent = DQNAgent(NUM_CHANNELS, 25, device=torch.device("cpu"), width=16, blocks=2, n_step=2)
    replay = PrioritizedReplay(200, (NUM_CHANNELS, 5, 5), 25)
    rng = np.random.default_rng(0)
    for i in range(64):
        s = rng.random((NUM_CHANNELS, 5, 5), dtype=np.float32)
        ns = rng.random((NUM_CHANNELS, 5, 5), dtype=np.float32)
        mask = (rng.random(25) > 0.3).astype(np.uint8)
        replay.add(s, int(i % 25), 0.1, ns, bool(i % 7 == 0), mask)
    batch = replay.sample(16, beta=0.5)
    loss, td = agent.learn(batch, batch["weights"])
    assert np.isfinite(loss)
    assert td.shape == (16,)
    assert np.all(np.isfinite(td))
    replay.update_priorities(batch["indices"], td)


def test_terminal_transition_no_nan():
    # All next-actions masked out (terminal) + done=True must not produce NaN in the target.
    agent = DQNAgent(NUM_CHANNELS, 25, device=torch.device("cpu"), width=8, blocks=1, n_step=1)
    replay = PrioritizedReplay(50, (NUM_CHANNELS, 5, 5), 25)
    zeros = np.zeros((NUM_CHANNELS, 5, 5), dtype=np.float32)
    for i in range(32):
        replay.add(zeros, i % 25, 1.0, zeros, True, np.zeros(25, dtype=np.uint8))
    batch = replay.sample(8, beta=0.5)
    loss, td = agent.learn(batch, batch["weights"])
    assert np.isfinite(loss) and np.all(np.isfinite(td))
