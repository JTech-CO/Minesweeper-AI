"""Visible-only count-aware posterior policy for M6-R4."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from trainer.encoding_v2 import VALID_CHANNEL
from trainer.models.policy_value import MASK_VALUE, _groups


class ChannelLayerNorm2d(nn.Module):
    """Normalize channels per cell so padded area cannot affect valid cells."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.permute(0, 2, 3, 1)
        return self.norm(value).permute(0, 3, 1, 2)


class MaskedGroupNorm2d(nn.Module):
    """GroupNorm over valid cells only; equals GroupNorm on unpadded boards."""

    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.groups = _groups(width)
        self.channels_per_group = width // self.groups
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, channels, rows, cols = value.shape
        grouped = value.reshape(
            batch,
            self.groups,
            self.channels_per_group,
            rows,
            cols,
        )
        mask = valid.reshape(batch, 1, 1, rows, cols)
        count = (mask.sum(dim=(3, 4), keepdim=True) * self.channels_per_group).clamp_min(1.0)
        mean = (grouped * mask).sum(dim=(2, 3, 4), keepdim=True) / count
        variance = (
            ((grouped - mean).square() * mask).sum(dim=(2, 3, 4), keepdim=True)
            / count
        )
        normalized = ((grouped - mean) * torch.rsqrt(variance + self.eps)).reshape(
            batch,
            channels,
            rows,
            cols,
        )
        weight = self.weight.view(1, channels, 1, 1)
        bias = self.bias.view(1, channels, 1, 1)
        return (normalized * weight + bias) * valid


class MaskedResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm1 = MaskedGroupNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
        self.norm2 = MaskedGroupNorm2d(width)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(value)
        residual = F.silu(self.norm1(residual, valid)) * valid
        residual = self.norm2(self.conv2(residual), valid) * valid
        return F.silu(value + self.scale.tanh() * residual) * valid


class CountAwareConstraintBlock(nn.Module):
    """Shared hidden-to-clue and clue-to-hidden factor-graph update."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.hidden_message = nn.Conv2d(width, width, 1)
        self.clue_meta = nn.Sequential(
            nn.Conv2d(4, width, 1),
            ChannelLayerNorm2d(width),
            nn.SiLU(),
        )
        self.clue_update = nn.Sequential(
            nn.Conv2d(width * 4, width, 1),
            ChannelLayerNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 1),
        )
        self.clue_message = nn.Conv2d(width, width, 1)
        self.hidden_update = nn.Sequential(
            nn.Conv2d(width * 3, width, 1),
            ChannelLayerNorm2d(width),
            nn.SiLU(),
            nn.Conv2d(width, width, 1),
        )
        self.clue_scale = nn.Parameter(torch.tensor(0.1))
        self.hidden_scale = nn.Parameter(torch.tensor(0.1))
        self.register_buffer(
            "neighbor_kernel",
            torch.ones(1, 1, 3, 3),
            persistent=False,
        )

    def _neighbor_sum(self, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        channels = value.shape[1]
        kernel = self.neighbor_kernel.expand(channels, 1, 3, 3)
        return F.conv2d(value * mask, kernel, padding=1, groups=channels)

    @staticmethod
    def _neighbor_max(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = torch.where(mask.bool(), value, torch.full_like(value, -1e4))
        pooled = F.max_pool2d(masked, 3, stride=1, padding=1)
        count = F.max_pool2d(mask, 3, stride=1, padding=1)
        return torch.where(count.bool(), pooled, torch.zeros_like(pooled))

    def forward(
        self,
        h: torch.Tensor,
        hidden: torch.Tensor,
        clue: torch.Tensor,
        clue_features: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        hidden_message = self.hidden_message(h)
        hidden_sum = self._neighbor_sum(hidden_message, hidden)
        hidden_max = self._neighbor_max(hidden_message, hidden)
        meta = self.clue_meta(clue_features)
        clue_delta = self.clue_update(
            torch.cat((h, hidden_sum, hidden_max, meta), dim=1)
        )
        h = h + self.clue_scale.tanh() * clue_delta * clue

        clue_message = self.clue_message(h + meta * clue)
        clue_sum = self._neighbor_sum(clue_message, clue)
        clue_max = self._neighbor_max(clue_message, clue)
        hidden_delta = self.hidden_update(torch.cat((h, clue_sum, clue_max), dim=1))
        h = h + self.hidden_scale.tanh() * hidden_delta * hidden
        return F.silu(h) * valid


class ConstraintPosteriorPolicyNet(nn.Module):
    """Count-aware policy with separate posterior and mine-supervision heads."""

    def __init__(
        self,
        in_channels: int,
        width: int = 128,
        blocks: int = 8,
        graph_rounds: int = 8,
        posterior_policy_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if graph_rounds < 1:
            raise ValueError("constraint-posterior-v5 requires graph_rounds >= 1")
        if posterior_policy_weight <= 0:
            raise ValueError("posterior_policy_weight must be positive")
        self.graph_rounds = graph_rounds
        self.stem_conv = nn.Conv2d(in_channels, width, 3, padding=1)
        self.stem_norm = MaskedGroupNorm2d(width)
        self.constraint_graph = CountAwareConstraintBlock(width)
        cycle = (1, 2, 4, 8, 4, 2, 1, 1)
        self.blocks = nn.ModuleList(
            MaskedResidualBlock(width, cycle[index % len(cycle)])
            for index in range(blocks)
        )
        self.global_proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU())
        self.preference_head = nn.Conv2d(width, 1, 1)
        self.posterior_head = nn.Conv2d(width, 1, 1)
        self.mine_head = nn.Conv2d(width, 1, 1)
        self.certainty_head = nn.Conv2d(width, 3, 1)
        self.posterior_policy_scale = nn.Parameter(torch.empty(()))
        self.set_posterior_policy_weight(posterior_policy_weight)
        self.value_head = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
            nn.Tanh(),
        )
        self.register_buffer(
            "neighbor_kernel",
            torch.ones(1, 1, 3, 3),
            persistent=False,
        )

    def set_posterior_policy_weight(self, weight: float) -> None:
        if weight <= 0:
            raise ValueError("posterior policy weight must be positive")
        inverse_softplus = math.log(math.expm1(weight))
        with torch.no_grad():
            self.posterior_policy_scale.fill_(inverse_softplus)

    def _visible_constraint_features(
        self,
        x: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = x[:, 0:1] * valid
        clue_planes = x[:, 1:10]
        clue = clue_planes.sum(dim=1, keepdim=True).clamp_max(1.0) * valid
        values = torch.arange(9, device=x.device, dtype=x.dtype).view(1, 9, 1, 1)
        clue_value = (clue_planes * values).sum(dim=1, keepdim=True)
        hidden_count = F.conv2d(hidden, self.neighbor_kernel, padding=1)
        # Reveal-only play has no flagged-mine input; this plane remains explicit
        # so a future visible flag channel can be wired without changing the block.
        known_mines = torch.zeros_like(clue_value)
        residual = (clue_value - known_mines).clamp_min(0.0)
        features = torch.cat(
            (clue_value, hidden_count, known_mines, residual),
            dim=1,
        ) / 8.0
        return hidden, clue, features * clue

    def _features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = x[:, VALID_CHANNEL : VALID_CHANNEL + 1]
        hidden, clue, clue_features = self._visible_constraint_features(x, valid)
        h = F.silu(self.stem_norm(self.stem_conv(x), valid)) * valid
        for _ in range(self.graph_rounds):
            h = self.constraint_graph(h, hidden, clue, clue_features, valid)
        for block in self.blocks:
            h = block(h, valid)

        denom = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        pooled = (h * valid).sum(dim=(2, 3), keepdim=True) / denom
        h = F.silu(h + self.global_proj(pooled)) * valid
        return h, pooled, valid

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h, pooled, valid = self._features(x)
        posterior_map = self.posterior_head(h) * valid
        mine_map = self.mine_head(h) * valid
        preference = self.preference_head(h)
        penalty = F.softplus(self.posterior_policy_scale) * torch.sigmoid(posterior_map)
        policy = (preference - penalty).flatten(start_dim=1)
        if legal_mask is not None:
            legal = legal_mask.flatten(start_dim=1).bool()
            policy = torch.where(legal, policy, torch.full_like(policy, MASK_VALUE))
        posterior = posterior_map.flatten(start_dim=1)
        return {
            "policy_logits": policy,
            "posterior_logits": posterior,
            "risk_logits": posterior,
            "mine_logits": mine_map.flatten(start_dim=1),
            "certainty_logits": self.certainty_head(h) * valid,
            "value": self.value_head(pooled.flatten(start_dim=1)).squeeze(1),
        }
