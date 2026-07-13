"""Global-context spatial policy/value model for all board sizes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from trainer.encoding_v2 import VALID_CHANNEL

MASK_VALUE = -1.0e9


def _groups(width: int) -> int:
    for groups in (8, 4, 2):
        if width % groups == 0:
            return groups
    return 1


class DilatedResBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        groups = _groups(width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class SpatialPolicyValueNet(nn.Module):
    def __init__(self, in_channels: int, width: int = 96, blocks: int = 6) -> None:
        super().__init__()
        groups = _groups(width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
        )
        dilation_cycle = (1, 2, 4, 8, 4, 2, 1)
        self.blocks = nn.ModuleList(
            DilatedResBlock(width, dilation_cycle[index % len(dilation_cycle)])
            for index in range(blocks)
        )
        self.global_proj = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=1),
            nn.SiLU(),
        )
        self.policy_head = nn.Conv2d(width, 1, kernel_size=1)
        self.risk_head = nn.Conv2d(width, 1, kernel_size=1)
        self.certainty_head = nn.Conv2d(width, 3, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        valid = x[:, VALID_CHANNEL : VALID_CHANNEL + 1]
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        denom = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        pooled_map = (h * valid).sum(dim=(2, 3), keepdim=True) / denom
        h = F.silu(h + self.global_proj(pooled_map))

        policy = self.policy_head(h).flatten(start_dim=1)
        if legal_mask is not None:
            legal = legal_mask.flatten(start_dim=1).bool()
            policy = torch.where(legal, policy, torch.full_like(policy, MASK_VALUE))
        risk = self.risk_head(h).flatten(start_dim=1)
        certainty = self.certainty_head(h)
        value = self.value_head(pooled_map.flatten(start_dim=1)).squeeze(1)
        return {
            "policy_logits": policy,
            "risk_logits": risk,
            "certainty_logits": certainty,
            "value": value,
        }
