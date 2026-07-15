from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from trainer.config import ExperimentConfig
from trainer.data.hard_teacher import generate_component_balanced_teacher_dataset
from trainer.encoding_v2 import legal_mask_v2
from trainer.encoding_v3 import NUM_CHANNELS_V3, encode_v3
from trainer.env import MinesweeperEnv
from trainer.models import build_policy_model
from trainer.pretrain import TeacherPretrainer
from trainer.pretrain_posterior import transfer_v4_weights
from trainer.storage import load_checkpoint, save_checkpoint


def _model() -> torch.nn.Module:
    torch.manual_seed(19)
    model = build_policy_model(
        {
            "architecture": "constraint-posterior-v5",
            "input_channels": NUM_CHANNELS_V3,
            "width": 16,
            "blocks": 1,
            "graph_rounds": 2,
        }
    )
    return model.eval()


def _opened_env() -> MinesweeperEnv:
    env = MinesweeperEnv(9, 9, 10, seed=71, first_click_policy="safe-area")
    env.reset()
    env.step(40)
    return env


def test_v5_config_and_factory_contract() -> None:
    config = ExperimentConfig(
        architecture="constraint-posterior-v5",
        graph_rounds=8,
        width=16,
        blocks=1,
    )
    model = build_policy_model(
        config.to_dict() | {"posterior_policy_weight": 0.75}
    )
    assert model.graph_rounds == 8
    assert torch.isclose(
        torch.nn.functional.softplus(model.posterior_policy_scale),
        torch.tensor(0.75),
    )
    assert model.posterior_head is not model.mine_head
    with pytest.raises(ValueError, match="graph_rounds"):
        ExperimentConfig(
            architecture="constraint-posterior-v5",
            graph_rounds=0,
        )


def test_v5_visible_state_is_independent_of_hidden_mine_layout() -> None:
    env = _opened_env()
    changed = deepcopy(env)
    hidden = np.flatnonzero(changed.revealed == 0)
    changed.mine_layout[hidden] = np.roll(changed.mine_layout[hidden], 1)
    assert not np.array_equal(env.mine_layout, changed.mine_layout)
    state = encode_v3(env)
    changed_state = encode_v3(changed)
    assert np.array_equal(state, changed_state)

    legal = torch.from_numpy(env.legal_action_mask().reshape(1, 9, 9)).bool()
    model = _model()
    with torch.no_grad():
        original = model(torch.from_numpy(state[None]), legal)["policy_logits"]
        mutated = model(torch.from_numpy(changed_state[None]), legal)["policy_logits"]
    assert torch.equal(original, mutated)


def test_v5_inference_does_not_call_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("solver must not run during policy inference")

    monkeypatch.setattr("trainer.solver.solve_step", forbidden)
    env = _opened_env()
    state = torch.from_numpy(encode_v3(env)[None])
    legal = torch.from_numpy(env.legal_action_mask().reshape(1, 9, 9)).bool()
    output = _model()(state, legal)
    assert output["policy_logits"].shape == (1, 81)


def test_v5_padding_invariance_and_illegal_masking() -> None:
    env = _opened_env()
    direct_state = torch.from_numpy(encode_v3(env)[None])
    padded_state = torch.from_numpy(encode_v3(env, (16, 30))[None])
    direct_legal = torch.from_numpy(legal_mask_v2(env)[None]).bool()
    padded_legal = torch.from_numpy(legal_mask_v2(env, (16, 30))[None]).bool()
    model = _model()
    with torch.no_grad():
        direct = model(direct_state, direct_legal)
        padded = model(padded_state, padded_legal)

    padded_policy = padded["policy_logits"].reshape(1, 16, 30)[:, :9, :9]
    assert torch.allclose(
        direct["policy_logits"].reshape(1, 9, 9),
        padded_policy,
        atol=2e-5,
    )
    for output in (direct, padded):
        assert all(torch.isfinite(value).all() for value in output.values())
    padded_grid = padded["policy_logits"].reshape(1, 16, 30)
    assert torch.all(padded_grid[:, 9:, :] < -1e8)
    assert torch.all(padded_grid[:, :, 9:] < -1e8)


def test_v5_policy_is_structurally_coupled_to_posterior() -> None:
    env = _opened_env()
    state = torch.from_numpy(encode_v3(env)[None])
    legal = torch.from_numpy(env.legal_action_mask().reshape(1, 9, 9)).bool()
    model = _model()
    with torch.no_grad():
        before = model(state, legal)["policy_logits"]
        model.posterior_head.bias.add_(5.0)
        after = model(state, legal)["policy_logits"]
    assert not torch.equal(before, after)


def test_v5_teacher_updates_separate_posterior_and_mine_heads() -> None:
    dataset = generate_component_balanced_teacher_dataset(
        5,
        5,
        3,
        boards=8,
        seed_base=810_000_000,
        max_states=8,
        oversample_factor=1,
    )
    model = _model()
    posterior_before = model.posterior_head.weight.detach().clone()
    mine_before = model.mine_head.weight.detach().clone()
    trainer = TeacherPretrainer(
        model,
        device=torch.device("cpu"),
        learning_rate=1e-3,
    )

    metrics = trainer.train_epoch(dataset, batch_size=8)

    assert {"posterior_nll", "posterior_brier", "mine"} <= metrics.keys()
    assert all(np.isfinite(value) for value in metrics.values())
    assert not torch.equal(posterior_before, model.posterior_head.weight)
    assert not torch.equal(mine_before, model.mine_head.weight)

def test_v5_transfers_required_v4_policy_parameters(tmp_path: Path) -> None:
    source = build_policy_model(
        {
            "architecture": "constraint-graph-v4",
            "graph_rounds": 2,
            "width": 16,
            "blocks": 1,
        }
    )
    checkpoint = tmp_path / "v4.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "checkpoint_id": "v4-parent",
        },
        checkpoint,
    )
    target = _model()

    parent_id, transferred = transfer_v4_weights(target, checkpoint)

    assert parent_id == "v4-parent"
    assert transferred > 10
    assert torch.equal(target.preference_head.weight, source.policy_head.weight)
    assert torch.equal(target.posterior_head.weight, source.risk_head.weight)
    assert torch.equal(target.mine_head.weight, source.risk_head.weight)


def test_v5_checkpoint_reload_is_bit_identical(tmp_path: Path) -> None:
    env = _opened_env()
    state = torch.from_numpy(encode_v3(env)[None])
    legal = torch.from_numpy(env.legal_action_mask().reshape(1, 9, 9)).bool()
    model = _model()
    with torch.no_grad():
        expected = model(state, legal)
    config = {
        "architecture": "constraint-posterior-v5",
        "graph_rounds": 2,
        "input_channels": NUM_CHANNELS_V3,
        "width": 16,
        "blocks": 1,
    }
    checkpoint = tmp_path / "v5.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        config=config,
        run_id="v5-roundtrip",
        git_sha="test",
        episode=1,
        global_step=1,
    )
    restored = _model()
    load_checkpoint(
        checkpoint,
        model=restored,
        expected_config=config,
        map_location="cpu",
    )
    restored.eval()
    with torch.no_grad():
        actual = restored(state, legal)

    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)