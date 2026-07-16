"""Ranking-aware teacher objectives and diagnostics for constraint-ranker-v6."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from trainer.models.policy_value import MASK_VALUE


@dataclass(frozen=True)
class RankerLossWeights:
    policy: float = 0.25
    hard_rank: float = 1.0
    risk_regret: float = 2.0
    guess_neutral: float = 0.5
    posterior_nll: float = 0.5
    posterior_brier: float = 0.5
    certainty: float = 0.75
    mine: float = 0.05
    value: float = 0.05


DEFAULT_RANKER_LOSS_WEIGHTS = RankerLossWeights()


def _hard_rank_loss(
    score: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    states: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    floor = torch.finfo(score.dtype).min
    active = states & positive.any(dim=1) & negative.any(dim=1)
    if not active.any():
        return score.sum() * 0.0
    positive_score = score.masked_fill(~positive, floor).amax(dim=1)
    negative_score = score.masked_fill(~negative, floor).amax(dim=1)
    return F.softplus(margin + negative_score[active] - positive_score[active]).mean()


def ranker_losses(
    output: dict[str, torch.Tensor],
    *,
    legal: torch.Tensor,
    target_policy: torch.Tensor,
    target_risk: torch.Tensor,
    target_mine: torch.Tensor,
    target_certainty: torch.Tensor,
    target_value: torch.Tensor,
    weights: RankerLossWeights = DEFAULT_RANKER_LOSS_WEIGHTS,
    rank_margin: float = 0.5,
    risk_temperature: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Train certainty and guess ranking on disjoint state classes."""

    if rank_margin < 0 or risk_temperature <= 0:
        raise ValueError("rank margin must be non-negative and temperature positive")
    policy_logits = output["policy_logits"].float()
    legal_flat = legal.flatten(start_dim=1).bool()
    policy_target = target_policy.flatten(start_dim=1).float()
    risk_target = target_risk.flatten(start_dim=1).float()
    mine_target = target_mine.flatten(start_dim=1).float()
    certainty_target = target_certainty.flatten(start_dim=1)

    target_safe = (certainty_target == 1) & legal_flat
    has_certain_safe = target_safe.any(dim=1)
    guess_state = ~has_certain_safe

    certainty_probability = torch.softmax(
        output["certainty_logits"].float(),
        dim=1,
    )
    safe_margin = (certainty_probability[:, 1] - certainty_probability[:, 2]).flatten(start_dim=1)
    certainty_rank = _hard_rank_loss(
        safe_margin,
        target_safe,
        legal_flat & ~target_safe,
        has_certain_safe,
        rank_margin,
    )

    posterior_flat = output["posterior_logits"].float()
    raw_posterior_score = -torch.sigmoid(posterior_flat)
    posterior_score = torch.where(
        legal_flat,
        raw_posterior_score,
        torch.full_like(raw_posterior_score, MASK_VALUE),
    )
    target_action = (policy_target > 0) & legal_flat
    posterior_rank = _hard_rank_loss(
        posterior_score,
        target_action,
        legal_flat & ~target_action,
        guess_state,
        rank_margin,
    )
    hard_rank = certainty_rank + posterior_rank

    combined_policy_loss = -(policy_target * F.log_softmax(policy_logits, dim=1)).sum(dim=1)
    guess_policy_loss = -(
        policy_target * F.log_softmax(posterior_score / risk_temperature, dim=1)
    ).sum(dim=1)
    policy_loss = torch.where(
        has_certain_safe,
        combined_policy_loss,
        guess_policy_loss,
    ).mean()

    posterior_probability = F.softmax(
        posterior_score / risk_temperature,
        dim=1,
    )
    legal_risk = risk_target.masked_fill(~legal_flat, float("inf"))
    minimum_risk = legal_risk.amin(dim=1)
    expected_risk = (posterior_probability * risk_target.masked_fill(~legal_flat, 0.0)).sum(dim=1)
    if guess_state.any():
        risk_regret = (expected_risk[guess_state] - minimum_risk[guess_state]).clamp_min(0.0).mean()
        guess_cells = guess_state[:, None] & legal_flat
        guess_neutral = safe_margin[guess_cells].square().mean()
    else:
        risk_regret = policy_logits.sum() * 0.0
        guess_neutral = policy_logits.sum() * 0.0

    posterior_logits = posterior_flat[legal_flat]
    legal_risk_target = risk_target[legal_flat]
    posterior_nll = F.binary_cross_entropy_with_logits(
        posterior_logits,
        legal_risk_target,
    )
    posterior_brier = F.mse_loss(
        torch.sigmoid(posterior_logits),
        legal_risk_target,
    )
    mine_loss = F.binary_cross_entropy_with_logits(
        output["mine_logits"].float()[legal_flat],
        mine_target[legal_flat],
    )
    class_weights = output["certainty_logits"].new_tensor((0.5, 1.0, 1.25))
    certainty_loss = F.cross_entropy(
        output["certainty_logits"].float(),
        target_certainty,
        weight=class_weights.float(),
        ignore_index=-100,
    )
    value_loss = F.mse_loss(output["value"].float(), target_value.float())

    total = (
        weights.policy * policy_loss
        + weights.hard_rank * hard_rank
        + weights.risk_regret * risk_regret
        + weights.guess_neutral * guess_neutral
        + weights.posterior_nll * posterior_nll
        + weights.posterior_brier * posterior_brier
        + weights.certainty * certainty_loss
        + weights.mine * mine_loss
        + weights.value * value_loss
    )
    return {
        "loss": total,
        "policy": policy_loss,
        "hard_rank": hard_rank,
        "certainty_rank": certainty_rank,
        "posterior_rank": posterior_rank,
        "risk_regret": risk_regret,
        "guess_neutral": guess_neutral,
        "posterior_nll": posterior_nll,
        "posterior_brier": posterior_brier,
        "mine": mine_loss,
        "certainty": certainty_loss,
        "value": value_loss,
    }


@torch.no_grad()
def ranker_metrics(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, float | int]:
    """Measure calibration plus the tail errors that terminate full games."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    nll_sum = 0.0
    brier_sum = 0.0
    legal_cells = 0
    predicted_safe = 0
    predicted_safe_actual = 0
    target_safe = 0
    recovered_safe = 0
    states = 0
    teacher_hits = 0
    top1_mines = 0
    top1_risk_sum = 0.0
    top1_regret_sum = 0.0
    certain_states = 0
    certain_actions = 0
    try:
        for batch in loader:
            state = batch["state"].to(device)
            legal = batch["legal_mask"].to(device).bool()
            target_policy = batch["policy_target"].to(device)
            target_risk = batch["risk_target"].to(device)
            target_mine = batch["mine_target"].to(device)
            target_certainty = batch["certainty_target"].to(device)
            output = model(state, legal)

            legal_flat = legal.flatten(start_dim=1)
            risk_flat = target_risk.flatten(start_dim=1)
            posterior = output["posterior_logits"][legal_flat].float()
            risk = risk_flat[legal_flat].float()
            nll_sum += float(
                F.binary_cross_entropy_with_logits(
                    posterior,
                    risk,
                    reduction="sum",
                ).item()
            )
            probability = torch.sigmoid(posterior)
            brier_sum += float((probability - risk).square().sum().item())
            legal_cells += int(legal_flat.sum().item())

            predicted = output["certainty_logits"].argmax(dim=1)
            predicted_mask = (predicted == 1) & legal
            target_mask = (target_certainty == 1) & legal
            predicted_safe += int(predicted_mask.sum().item())
            predicted_safe_actual += int((predicted_mask & (target_mine == 0)).sum().item())
            target_safe += int(target_mask.sum().item())
            recovered_safe += int((predicted_mask & target_mask).sum().item())

            action = output["policy_logits"].argmax(dim=1)
            target_policy_flat = target_policy.flatten(start_dim=1)
            mine_flat = target_mine.flatten(start_dim=1)
            certainty_flat = target_certainty.flatten(start_dim=1)
            chosen_policy = target_policy_flat.gather(1, action[:, None]).squeeze(1)
            chosen_mine = mine_flat.gather(1, action[:, None]).squeeze(1)
            chosen_risk = risk_flat.gather(1, action[:, None]).squeeze(1)
            minimum_risk = risk_flat.masked_fill(~legal_flat, float("inf")).amin(dim=1)
            has_certain = target_mask.flatten(start_dim=1).any(dim=1)
            chosen_certainty = certainty_flat.gather(
                1,
                action[:, None],
            ).squeeze(1)

            count = state.shape[0]
            states += count
            teacher_hits += int((chosen_policy > 0).sum().item())
            top1_mines += int((chosen_mine > 0.5).sum().item())
            top1_risk_sum += float(chosen_risk.sum().item())
            top1_regret_sum += float((chosen_risk - minimum_risk).clamp_min(0.0).sum().item())
            certain_states += int(has_certain.sum().item())
            certain_actions += int(((chosen_certainty == 1) & has_certain).sum().item())
    finally:
        model.train(was_training)

    cell_denominator = max(legal_cells, 1)
    state_denominator = max(states, 1)
    return {
        "states": states,
        "legal_cells": legal_cells,
        "posterior_nll": nll_sum / cell_denominator,
        "posterior_brier": brier_sum / cell_denominator,
        "certain_safe_precision": predicted_safe_actual / max(predicted_safe, 1),
        "certain_safe_recall": recovered_safe / max(target_safe, 1),
        "certain_safe_predictions": predicted_safe,
        "certain_safe_targets": target_safe,
        "top1_teacher_accuracy": teacher_hits / state_denominator,
        "top1_mine_rate": top1_mines / state_denominator,
        "top1_risk": top1_risk_sum / state_denominator,
        "top1_regret": top1_regret_sum / state_denominator,
        "certain_safe_top1_rate": certain_actions / max(certain_states, 1),
        "certain_safe_states": certain_states,
    }
