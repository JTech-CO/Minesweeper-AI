"""Prioritized Experience Replay (Schaul et al.) — proportional variant with a sum-tree.

Stores encoded transitions and samples them in proportion to |TD-error|^alpha, returning
importance-sampling weights to correct the bias. (기술백서 §4.3)
"""

from __future__ import annotations

import numpy as np


class SumTree:
    """Fixed-capacity sum-tree: leaves hold priorities, internal nodes hold subtree sums."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_index: int, priority: float) -> None:
        leaf = data_index + self.capacity - 1
        change = priority - self.tree[leaf]
        self.tree[leaf] = priority
        parent = (leaf - 1) // 2
        while True:
            self.tree[parent] += change
            if parent == 0:
                break
            parent = (parent - 1) // 2

    def get(self, value: float) -> tuple[int, float]:
        """Return (data_index, priority) for the leaf whose cumulative range contains `value`."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):  # idx is a leaf
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1), float(self.tree[idx])


class PrioritizedReplay:
    def __init__(
        self,
        capacity: int,
        state_shape: tuple[int, ...],
        n_actions: int,
        *,
        alpha: float = 0.6,
        eps: float = 1e-5,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.eps = eps
        self.n_actions = n_actions
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self.size = 0
        self.pos = 0

        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.next_masks = np.zeros((capacity, n_actions), dtype=np.uint8)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        i = self.pos
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done
        self.next_masks[i] = next_mask
        # New transitions get max priority so they are sampled at least once.
        self.tree.update(i, self.max_priority**self.alpha)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> dict:
        idxs = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        total = self.tree.total()
        segment = total / batch_size
        for b in range(batch_size):
            value = np.random.uniform(segment * b, segment * (b + 1))
            data_index, priority = self.tree.get(value)
            # Guard against rare empty-range hits before the buffer fills.
            if data_index >= self.size:
                data_index = np.random.randint(self.size)
                priority = self.tree.tree[data_index + self.capacity - 1]
            idxs[b] = data_index
            priorities[b] = max(priority, self.eps)

        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()
        return {
            "states": self.states[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_states": self.next_states[idxs],
            "dones": self.dones[idxs],
            "next_masks": self.next_masks[idxs],
            "weights": weights.astype(np.float32),
            "indices": idxs,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for data_index, td in zip(indices, td_errors, strict=True):
            priority = (abs(float(td)) + self.eps) ** self.alpha
            self.tree.update(int(data_index), priority)
            self.max_priority = max(self.max_priority, abs(float(td)) + self.eps)
