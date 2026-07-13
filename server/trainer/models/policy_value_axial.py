"""Width-128 policy/value network with full-board axial context."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from trainer.encoding_v2 import VALID_CHANNEL
from trainer.models.policy_value import MASK_VALUE, DilatedResBlock, _groups


class AxialContextBlock(nn.Module):
    def __init__(self, width: int, heads: int = 4) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by attention heads")
        self.row_norm = nn.LayerNorm(width)
        self.row_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.col_norm = nn.LayerNorm(width)
        self.col_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(width * 2, width, kernel_size=1),
        )
        self.ffn_norm = nn.GroupNorm(_groups(width), width)

    @staticmethod
    def _safe_padding_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_padding = mask.all(dim=1)
        safe = mask.clone()
        if all_padding.any():
            safe[all_padding, 0] = False
        return safe, all_padding

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, channels, rows, cols = x.shape

        row_tokens = x.permute(0, 2, 3, 1).reshape(batch * rows, cols, channels)
        row_padding = ~valid[:, 0].reshape(batch * rows, cols).bool()
        row_padding, empty_rows = self._safe_padding_mask(row_padding)
        row_input = self.row_norm(row_tokens)
        row_out, _ = self.row_attention(
            row_input,
            row_input,
            row_input,
            key_padding_mask=row_padding,
            need_weights=False,
        )
        row_out[empty_rows] = 0
        row_tokens = row_tokens + row_out
        x = row_tokens.reshape(batch, rows, cols, channels).permute(0, 3, 1, 2)

        col_tokens = x.permute(0, 3, 2, 1).reshape(batch * cols, rows, channels)
        col_padding = ~valid[:, 0].permute(0, 2, 1).reshape(batch * cols, rows).bool()
        col_padding, empty_cols = self._safe_padding_mask(col_padding)
        col_input = self.col_norm(col_tokens)
        col_out, _ = self.col_attention(
            col_input,
            col_input,
            col_input,
            key_padding_mask=col_padding,
            need_weights=False,
        )
        col_out[empty_cols] = 0
        col_tokens = col_tokens + col_out
        x = col_tokens.reshape(batch, cols, rows, channels).permute(0, 3, 2, 1)
        return (x + self.ffn(F.silu(self.ffn_norm(x)))) * valid


class AxialPolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int = 128,
        blocks: int = 8,
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        groups = _groups(width)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
        )
        dilation_cycle = (1, 2, 4, 8, 4, 2, 1, 1)
        self.blocks = nn.ModuleList(
            DilatedResBlock(width, dilation_cycle[index % len(dilation_cycle)])
            for index in range(blocks)
        )
        attention_points = sorted({blocks // 2 - 1, blocks - 1})
        self.attention_points = set(attention_points)
        self.axial = nn.ModuleList(
            AxialContextBlock(width, attention_heads) for _ in attention_points
        )
        self.global_proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.SiLU())
        self.policy_head = nn.Conv2d(width, 1, 1)
        self.risk_head = nn.Conv2d(width, 1, 1)
        self.certainty_head = nn.Conv2d(width, 3, 1)
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
        h = self.stem(x) * valid
        attention_index = 0
        for index, block in enumerate(self.blocks):
            h = block(h) * valid
            if index in self.attention_points:
                h = self.axial[attention_index](h, valid)
                attention_index += 1

        denom = valid.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
        pooled = (h * valid).sum(dim=(2, 3), keepdim=True) / denom
        h = F.silu(h + self.global_proj(pooled)) * valid
        policy = self.policy_head(h).flatten(start_dim=1)
        if legal_mask is not None:
            legal = legal_mask.flatten(start_dim=1).bool()
            policy = torch.where(legal, policy, torch.full_like(policy, MASK_VALUE))
        return {
            "policy_logits": policy,
            "risk_logits": self.risk_head(h).flatten(start_dim=1),
            "certainty_logits": self.certainty_head(h),
            "value": self.value_head(pooled.flatten(start_dim=1)).squeeze(1),
        }

