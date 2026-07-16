"""Certainty-first risk ranker for M6-R5."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from trainer.models.policy_value import MASK_VALUE
from trainer.models.policy_value_posterior import ConstraintPosteriorPolicyNet


class ConstraintRankerPolicyNet(ConstraintPosteriorPolicyNet):
    """Rank legal cells by certainty, calibrated risk, then a bounded tie-breaker."""

    def __init__(
        self,
        in_channels: int,
        width: int = 128,
        blocks: int = 8,
        graph_rounds: int = 8,
        posterior_policy_weight: float = 1.0,
        certainty_policy_weight: float = 4.0,
        preference_policy_limit: float = 0.0,
    ) -> None:
        super().__init__(
            in_channels,
            width,
            blocks,
            graph_rounds,
            posterior_policy_weight,
        )
        if certainty_policy_weight <= 0:
            raise ValueError("certainty policy weight must be positive")
        if not 0 <= preference_policy_limit <= 0.25:
            raise ValueError("preference policy limit must be in [0, 0.25]")
        if posterior_policy_weight + 2 * preference_policy_limit >= certainty_policy_weight:
            raise ValueError("certainty weight must dominate risk and preference")
        inverse_softplus = math.log(math.expm1(certainty_policy_weight))
        self.certainty_policy_scale = torch.nn.Parameter(torch.tensor(inverse_softplus))
        self.register_buffer(
            "preference_policy_limit",
            torch.tensor(preference_policy_limit),
        )

    def _compose_policy(
        self,
        preference: torch.Tensor,
        posterior_map: torch.Tensor,
        certainty_map: torch.Tensor,
    ) -> torch.Tensor:
        certainty = torch.softmax(certainty_map, dim=1)
        soft_margin = certainty[:, 1:2] - certainty[:, 2:3]
        predicted_class = certainty_map.argmax(dim=1, keepdim=True)
        hard_margin = (predicted_class == 1).to(certainty_map.dtype)
        hard_margin = hard_margin - (predicted_class == 2).to(certainty_map.dtype)
        # Exact class priority in forward, soft certainty gradient in backward.
        certainty_priority = hard_margin + soft_margin - soft_margin.detach()
        tie_span = 2 * self.preference_policy_limit
        certainty_scale = F.softplus(self.certainty_policy_scale).clamp_min(tie_span + 1e-3)
        maximum_risk_scale = (certainty_scale - tie_span - 1e-3).clamp_min(0.0)
        risk_scale = torch.minimum(
            F.softplus(self.posterior_policy_scale),
            maximum_risk_scale,
        )
        safe_score = certainty_scale * certainty_priority
        risk_penalty = risk_scale * torch.sigmoid(posterior_map)
        tie_breaker = self.preference_policy_limit * torch.tanh(preference)
        return safe_score - risk_penalty + tie_breaker

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h, pooled, valid = self._features(x)
        posterior_map = self.posterior_head(h) * valid
        mine_map = self.mine_head(h) * valid
        certainty_map = self.certainty_head(h) * valid
        policy = self._compose_policy(
            self.preference_head(h),
            posterior_map,
            certainty_map,
        ).flatten(start_dim=1)
        if legal_mask is not None:
            legal = legal_mask.flatten(start_dim=1).bool()
            policy = torch.where(legal, policy, torch.full_like(policy, MASK_VALUE))
        posterior = posterior_map.flatten(start_dim=1)
        return {
            "policy_logits": policy,
            "posterior_logits": posterior,
            "risk_logits": posterior,
            "mine_logits": mine_map.flatten(start_dim=1),
            "certainty_logits": certainty_map,
            "value": self.value_head(pooled.flatten(start_dim=1)).squeeze(1),
        }
