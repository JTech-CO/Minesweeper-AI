"""Architecture-aware construction for policy checkpoints and workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trainer.encoding_v3 import NUM_CHANNELS_V3
from trainer.models.policy_value_axial import AxialPolicyValueNet


def build_policy_model(config: Mapping[str, Any]) -> AxialPolicyValueNet:
    architecture = str(config.get("architecture", "axial-v3"))
    if architecture not in {"axial-v3", "constraint-graph-v4"}:
        raise ValueError(f"unsupported policy architecture {architecture!r}")
    default_rounds = 4 if architecture == "constraint-graph-v4" else 0
    graph_rounds = int(config.get("graph_rounds", default_rounds))
    if architecture == "axial-v3" and graph_rounds:
        raise ValueError("axial-v3 checkpoint cannot enable graph rounds")
    return AxialPolicyValueNet(
        int(config.get("input_channels", NUM_CHANNELS_V3)),
        int(config.get("width", 128)),
        int(config.get("blocks", 8)),
        graph_rounds=graph_rounds,
    )
