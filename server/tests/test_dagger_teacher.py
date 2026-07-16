from __future__ import annotations

import numpy as np
import torch

from trainer.data.dagger_teacher import generate_on_policy_teacher_dataset


class LastLegalPolicy(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        legal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        count = state.shape[2] * state.shape[3]
        score = torch.arange(count, device=state.device).expand(state.shape[0], -1)
        score = torch.where(
            legal.flatten(start_dim=1),
            score,
            torch.full_like(score, -1e9),
        )
        return {"policy_logits": score}


def _decision(view, *, last: bool) -> dict:
    covered = np.flatnonzero(~view.revealed)
    cell = int(covered[-1 if last else 0])
    return {
        "safe": [cell],
        "mines": [],
        "guess": None,
        "probabilities": None,
    }


def test_dagger_solver_labels_do_not_change_policy_trajectory(monkeypatch) -> None:
    model = LastLegalPolicy()
    monkeypatch.setattr(
        "trainer.data.dagger_teacher.solve_step",
        lambda view: _decision(view, last=False),
    )
    first = generate_on_policy_teacher_dataset(
        model,
        device=torch.device("cpu"),
        rows=5,
        cols=5,
        mines=3,
        boards=2,
        seed_base=44,
        max_states=4,
        batch_size=2,
    )
    monkeypatch.setattr(
        "trainer.data.dagger_teacher.solve_step",
        lambda view: _decision(view, last=True),
    )
    second = generate_on_policy_teacher_dataset(
        model,
        device=torch.device("cpu"),
        rows=5,
        cols=5,
        mines=3,
        boards=2,
        seed_base=44,
        max_states=4,
        batch_size=2,
    )

    assert len(first) == len(second)
    assert all(
        np.array_equal(left.state, right.state)
        for left, right in zip(first.samples, second.samples, strict=True)
    )
    assert any(
        not np.array_equal(left.policy_target, right.policy_target)
        for left, right in zip(first.samples, second.samples, strict=True)
    )


def test_dagger_samples_are_compact_but_batches_expand_to_float32(monkeypatch) -> None:
    monkeypatch.setattr(
        "trainer.data.dagger_teacher.solve_step",
        lambda view: _decision(view, last=False),
    )
    dataset = generate_on_policy_teacher_dataset(
        LastLegalPolicy(),
        device=torch.device("cpu"),
        rows=5,
        cols=5,
        mines=3,
        boards=1,
        seed_base=51,
        max_states=2,
        batch_size=1,
    )

    assert dataset.samples[0].state.dtype == np.float16
    assert dataset.samples[0].mine_target.dtype == np.uint8
    item = dataset[0]
    assert item["state"].dtype == torch.float32
    assert item["policy_target"].dtype == torch.float32
    assert item["certainty_target"].dtype == torch.int64
