"""Architecture-aware construction for policy checkpoints and workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.models.policy_value_axial import AxialPolicyValueNet
from trainer.models.policy_value_posterior import ConstraintPosteriorPolicyNet
from trainer.models.policy_value_ranker import ConstraintRankerPolicyNet


def build_policy_model(
    config: Mapping[str, Any],
) -> (
    AxialPolicyValueNet
    | ConstraintPosteriorPolicyNet
    | ConstraintRankerPolicyNet
):
    architecture = str(config.get("architecture", "axial-v3"))
    if architecture not in {
        "axial-v3",
        "constraint-graph-v4",
        "constraint-posterior-v5",
        "constraint-ranker-v6",
    }:
        raise ValueError(f"unsupported policy architecture {architecture!r}")
    default_rounds = {
        "axial-v3": 0,
        "constraint-graph-v4": 4,
        "constraint-posterior-v5": 8,
        "constraint-ranker-v6": 8,
    }[architecture]
    graph_rounds = int(config.get("graph_rounds", default_rounds))
    if architecture == "axial-v3" and graph_rounds:
        raise ValueError("axial-v3 checkpoint cannot enable graph rounds")
    if architecture == "constraint-ranker-v6":
        return ConstraintRankerPolicyNet(
            int(config.get("input_channels", NUM_CHANNELS_V3)),
            int(config.get("width", 128)),
            int(config.get("blocks", 8)),
            graph_rounds=graph_rounds,
            posterior_policy_weight=float(
                config.get("posterior_policy_weight", 1.0)
            ),
            certainty_policy_weight=float(
                config.get("certainty_policy_weight", 4.0)
            ),
            preference_policy_limit=float(
                config.get("preference_policy_limit", 0.0)
            ),
        )
    if architecture == "constraint-posterior-v5":
        return ConstraintPosteriorPolicyNet(
            int(config.get("input_channels", NUM_CHANNELS_V3)),
            int(config.get("width", 128)),
            int(config.get("blocks", 8)),
            graph_rounds=graph_rounds,
            posterior_policy_weight=float(
                config.get("posterior_policy_weight", 0.5)
            ),
        )
    return AxialPolicyValueNet(
        int(config.get("input_channels", NUM_CHANNELS_V3)),
        int(config.get("width", 128)),
        int(config.get("blocks", 8)),
        graph_rounds=graph_rounds,
    )
