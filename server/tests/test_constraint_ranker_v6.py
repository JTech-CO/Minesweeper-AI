from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from trainer.config import ExperimentConfig
from trainer.encoding_v2 import legal_mask_v2
from trainer.encoding_v3 import NUM_CHANNELS_V3, encode_v3
from trainer.env import MinesweeperEnv
from trainer.models import build_policy_model
from trainer.pretrain_ranker import transfer_v5_weights
from trainer.ranker_loss import ranker_losses
from trainer.storage import load_checkpoint, save_checkpoint


def _config() -> dict:
    return {
        "architecture": "constraint-ranker-v6",
        "input_channels": NUM_CHANNELS_V3,
        "width": 16,
        "blocks": 1,
        "graph_rounds": 2,
        "posterior_policy_weight": 1.0,
        "certainty_policy_weight": 4.0,
        "preference_policy_limit": 0.05,
    }


def _model() -> torch.nn.Module:
    torch.manual_seed(41)
    return build_policy_model(_config()).eval()


def _opened_env() -> MinesweeperEnv:
    env = MinesweeperEnv(9, 9, 10, seed=73, first_click_policy="safe-area")
    env.reset()
    env.step(40)
    return env


def test_v6_config_factory_and_scale_contract() -> None:
    config = ExperimentConfig(
        architecture="constraint-ranker-v6",
        graph_rounds=8,
        width=16,
        blocks=1,
    )
    model = build_policy_model(
        config.to_dict()
        | {
            "certainty_policy_weight": 3.0,
            "posterior_policy_weight": 0.75,
            "preference_policy_limit": 0.02,
        }
    )
    assert torch.isclose(
        F.softplus(model.certainty_policy_scale),
        torch.tensor(3.0),
    )
    assert torch.isclose(
        F.softplus(model.posterior_policy_scale),
        torch.tensor(0.75),
    )
    assert torch.isclose(
        model.preference_policy_limit,
        torch.tensor(0.02),
    )
    with pytest.raises(ValueError, match="graph_rounds"):
        ExperimentConfig(
            architecture="constraint-ranker-v6",
            graph_rounds=0,
        )
    with pytest.raises(ValueError, match="must dominate"):
        build_policy_model(
            _config()
            | {
                "posterior_policy_weight": 4.0,
                "certainty_policy_weight": 4.0,
            }
        )


def test_v6_class_priority_cannot_be_overridden_by_risk() -> None:
    model = _model()
    with torch.no_grad():
        model.certainty_policy_scale.fill_(-20.0)
        model.posterior_policy_scale.fill_(20.0)
    preference = torch.zeros((1, 1, 1, 3))
    posterior = torch.tensor([[[[10.0, -10.0, -10.0]]]])
    certainty = torch.tensor(
        [
            [
                [[0.0, 10.0, 0.0]],
                [[10.0, 0.0, 0.0]],
                [[0.0, 0.0, 10.0]],
            ]
        ],
        requires_grad=True,
    )

    score = model._compose_policy(preference, posterior, certainty)

    assert score[0, 0, 0, 0] > score[0, 0, 0, 1]
    assert score[0, 0, 0, 1] > score[0, 0, 0, 2]
    score.sum().backward()
    assert certainty.grad is not None
    assert torch.count_nonzero(certainty.grad)


def test_v6_visible_only_padding_and_solver_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("solver must not run during policy inference")

    monkeypatch.setattr("trainer.solver.solve_step", forbidden)
    env = _opened_env()
    changed = deepcopy(env)
    hidden = np.flatnonzero(changed.revealed == 0)
    changed.mine_layout[hidden] = np.roll(changed.mine_layout[hidden], 1)
    assert np.array_equal(encode_v3(env), encode_v3(changed))

    direct_state = torch.from_numpy(encode_v3(env)[None])
    padded_state = torch.from_numpy(encode_v3(env, (16, 30))[None])
    direct_legal = torch.from_numpy(legal_mask_v2(env)[None]).bool()
    padded_legal = torch.from_numpy(legal_mask_v2(env, (16, 30))[None]).bool()
    model = _model()
    with torch.no_grad():
        direct = model(direct_state, direct_legal)
        padded = model(padded_state, padded_legal)

    assert torch.allclose(
        direct["policy_logits"].reshape(1, 9, 9),
        padded["policy_logits"].reshape(1, 16, 30)[:, :9, :9],
        atol=2e-5,
    )
    assert torch.all(padded["policy_logits"].reshape(1, 16, 30)[:, 9:, :] < -1e8)


def test_v6_preference_is_a_bounded_tie_breaker() -> None:
    env = _opened_env()
    state = torch.from_numpy(encode_v3(env)[None])
    legal = torch.from_numpy(legal_mask_v2(env)[None]).bool()
    model = _model()
    with torch.no_grad():
        model.preference_head.bias.fill_(-100.0)
        low = model(state, legal)["policy_logits"]
        model.preference_head.bias.fill_(100.0)
        high = model(state, legal)["policy_logits"]
    difference = (high - low)[legal.flatten(start_dim=1)]
    assert torch.all(difference >= 0)
    assert float(difference.max()) <= 0.10001


def test_v6_transfers_complete_v5_trunk_and_heads(tmp_path: Path) -> None:
    source_config = _config() | {"architecture": "constraint-posterior-v5"}
    source_config.pop("certainty_policy_weight")
    source_config.pop("preference_policy_limit")
    source = build_policy_model(source_config)
    checkpoint = tmp_path / "v5.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "config": source_config,
            "checkpoint_id": "v5-parent",
        },
        checkpoint,
    )
    target = _model()

    parent_id, transferred, _ = transfer_v5_weights(target, checkpoint)

    assert parent_id == "v5-parent"
    assert transferred == len(source.state_dict())
    assert torch.equal(target.stem_conv.weight, source.stem_conv.weight)
    assert torch.equal(target.certainty_head.weight, source.certainty_head.weight)
    assert torch.equal(target.posterior_head.weight, source.posterior_head.weight)


def _loss_output(policy_logits: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "policy_logits": policy_logits,
        "posterior_logits": torch.tensor([[0.0, 0.0, 0.0]]),
        "mine_logits": torch.tensor([[0.0, 0.0, 0.0]]),
        "certainty_logits": torch.tensor(
            [[[[0.0, 0.0, 0.0]], [[2.0, 0.0, -1.0]], [[-1.0, 0.0, 2.0]]]]
        ),
        "value": torch.tensor([0.0]),
    }


def test_certain_state_loss_uses_combined_policy() -> None:
    targets = {
        "legal": torch.ones((1, 1, 3), dtype=torch.bool),
        "target_policy": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "target_risk": torch.tensor([[[0.0, 0.5, 1.0]]]),
        "target_mine": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "target_certainty": torch.tensor([[[1, 0, 2]]]),
        "target_value": torch.tensor([0.0]),
    }
    good = ranker_losses(
        _loss_output(torch.tensor([[2.0, 0.0, -1.0]])),
        **targets,
    )
    bad = ranker_losses(
        _loss_output(torch.tensor([[-1.0, 0.0, 2.0]])),
        **targets,
    )

    assert good["policy"] < bad["policy"]
    assert good["loss"] < bad["loss"]


def test_guess_state_loss_ranks_posterior_without_certainty_drift() -> None:
    targets = {
        "legal": torch.ones((1, 1, 3), dtype=torch.bool),
        "target_policy": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "target_risk": torch.tensor([[[0.0, 0.5, 1.0]]]),
        "target_mine": torch.tensor([[[0.0, 0.0, 1.0]]]),
        "target_certainty": torch.tensor([[[0, 0, 0]]]),
        "target_value": torch.tensor([0.0]),
    }
    good_output = _loss_output(torch.zeros((1, 3)))
    good_output["posterior_logits"] = torch.tensor([[-3.0, 0.0, 3.0]])
    bad_output = _loss_output(torch.zeros((1, 3)))
    bad_output["posterior_logits"] = torch.tensor([[3.0, 0.0, -3.0]])

    good = ranker_losses(good_output, **targets)
    bad = ranker_losses(bad_output, **targets)

    assert good["posterior_rank"] < bad["posterior_rank"]
    assert good["risk_regret"] < bad["risk_regret"]
    assert good["loss"] < bad["loss"]


def test_guess_loss_ignores_illegal_posterior_cells() -> None:
    targets = {
        "legal": torch.tensor([[[1, 1, 1, 0]]], dtype=torch.bool),
        "target_policy": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        "target_risk": torch.tensor([[[0.0, 0.5, 1.0, 0.0]]]),
        "target_mine": torch.tensor([[[0.0, 0.0, 1.0, 0.0]]]),
        "target_certainty": torch.tensor([[[0, 0, 0, -100]]]),
        "target_value": torch.tensor([0.0]),
    }

    def output(illegal_posterior: float) -> dict[str, torch.Tensor]:
        certainty = torch.zeros((1, 3, 1, 4))
        certainty[:, 0] = 2.0
        return {
            "policy_logits": torch.tensor([[0.0, 0.0, 0.0, -1e9]]),
            "posterior_logits": torch.tensor([[-3.0, 0.0, 3.0, illegal_posterior]]),
            "mine_logits": torch.zeros((1, 4)),
            "certainty_logits": certainty,
            "value": torch.tensor([0.0]),
        }

    low = ranker_losses(output(-100.0), **targets)
    high = ranker_losses(output(100.0), **targets)

    for name in ("loss", "policy", "posterior_rank", "risk_regret"):
        assert torch.equal(low[name], high[name])


def test_v6_checkpoint_reload_is_bit_identical(tmp_path: Path) -> None:
    env = _opened_env()
    state = torch.from_numpy(encode_v3(env)[None])
    legal = torch.from_numpy(legal_mask_v2(env)[None]).bool()
    model = _model()
    with torch.no_grad():
        expected = model(state, legal)
    checkpoint = tmp_path / "v6.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        config=_config(),
        run_id="v6-roundtrip",
        git_sha="test",
        episode=1,
        global_step=1,
    )
    restored = _model()
    load_checkpoint(
        checkpoint,
        model=restored,
        expected_config=_config(),
        map_location="cpu",
    )
    restored.eval()
    with torch.no_grad():
        actual = restored(state, legal)

    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)
