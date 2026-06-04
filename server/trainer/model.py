"""Fully-convolutional Q-network (기술백서 §4.3).

No fully-connected layers → the same weights apply to any board size, enabling the
Beginner→Intermediate→Expert curriculum transfer (M6). The head is a 1×1 conv that emits
one Q-value per cell; the H×W output is flattened to the action space. Action masking is
applied OUTSIDE the model (run-book: keep masking external and consistent).

BatchNorm is intentionally omitted — its train/eval running-stat mismatch is a common
source of "runs but doesn't learn" bugs in DQN with tiny eval batches.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class QNet(nn.Module):
    def __init__(self, in_channels: int, width: int = 64, blocks: int = 4) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(blocks)])
        self.head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) → per-cell Q flattened to (B, H*W)."""
        h = F.relu(self.stem(x))
        for block in self.blocks:
            h = block(h)
        q = self.head(h)  # (B, 1, H, W)
        return q.flatten(start_dim=1)  # (B, H*W)
