"""Stable inference rule for the M4-R2 multi-head model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from trainer.models import SpatialPolicyValueNet


@dataclass
class SpatialInferencePolicy:
    model: SpatialPolicyValueNet
    device: torch.device
    certainty_weight: float = 2.0
    risk_weight: float = 3.0

    @torch.no_grad()
    def act(self, state: np.ndarray, legal_mask: np.ndarray) -> int:
        x = torch.from_numpy(state[None]).to(self.device)
        legal = torch.from_numpy(legal_mask[None]).to(self.device).bool()
        output = self.model(x, legal)
        safe = output["certainty_logits"][:, 1].flatten(start_dim=1)
        mine = output["certainty_logits"][:, 2].flatten(start_dim=1)
        score = (
            output["policy_logits"]
            + self.certainty_weight * (safe - mine)
            - self.risk_weight * output["risk_logits"]
        )
        legal_flat = legal.flatten(start_dim=1)
        score = torch.where(legal_flat, score, torch.full_like(score, -1.0e9))
        return int(score.argmax(dim=1).item())
