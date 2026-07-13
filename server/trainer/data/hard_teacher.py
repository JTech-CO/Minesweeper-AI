"""Hard-state and frontier-component balanced solver-teacher data."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace

import numpy as np

from trainer.data.teacher import TeacherDataset, TeacherSample, generate_teacher_episode
from trainer.encoding_v3 import upgrade_v2_to_v3


@dataclass(frozen=True)
class StateProfile:
    guess: bool
    components: int
    progress_bin: int
    hardness: float

    @property
    def bucket(self) -> tuple[int, int, int]:
        return (int(self.guess), min(self.components, 4), self.progress_bin)


def _component_count(frontier: np.ndarray) -> int:
    rows, cols = frontier.shape
    unseen = {tuple(int(value) for value in point) for point in np.argwhere(frontier)}
    components = 0
    while unseen:
        components += 1
        queue = deque([unseen.pop()])
        while queue:
            row, col = queue.popleft()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbor = (row + dr, col + dc)
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        queue.append(neighbor)
    return components


def profile_sample(sample: TeacherSample) -> StateProfile:
    legal = sample.legal_mask.astype(bool)
    certain_safe = (sample.certainty_target == 1) & legal
    probabilities = np.clip(sample.risk_target[legal], 1e-6, 1 - 1e-6)
    entropy = (
        float(
            np.mean(
                -probabilities * np.log(probabilities)
                - (1 - probabilities) * np.log(1 - probabilities)
            )
        )
        if probabilities.size
        else 0.0
    )
    components = _component_count(sample.state[13] > 0.5)
    progress = float(sample.state[12, 0, 0])
    guess = not bool(certain_safe.any())
    hardness = 4.0 * guess + 0.75 * min(components, 6) + progress + entropy
    return StateProfile(
        guess=guess,
        components=components,
        progress_bin=min(3, int(progress * 4)),
        hardness=hardness,
    )


def balance_teacher_samples(
    candidates: list[TeacherSample],
    *,
    max_states: int,
) -> list[TeacherSample]:
    buckets: dict[tuple[int, int, int], list[tuple[float, TeacherSample]]] = defaultdict(list)
    for sample in candidates:
        profile = profile_sample(sample)
        buckets[profile.bucket].append((profile.hardness, sample))
    for values in buckets.values():
        values.sort(key=lambda item: item[0], reverse=True)

    selected: list[TeacherSample] = []
    active = sorted(buckets)
    while active and len(selected) < max_states:
        next_active = []
        for key in active:
            values = buckets[key]
            if values and len(selected) < max_states:
                _, sample = values.pop(0)
                selected.append(replace(sample, state=upgrade_v2_to_v3(sample.state)))
            if values:
                next_active.append(key)
        active = next_active
    return selected


def generate_component_balanced_teacher_dataset(
    rows: int,
    cols: int,
    mines: int,
    *,
    boards: int,
    seed_base: int,
    max_states: int,
    oversample_factor: int = 3,
) -> TeacherDataset:
    candidates: list[TeacherSample] = []
    target = max_states * max(1, oversample_factor)
    for board in range(boards):
        candidates.extend(generate_teacher_episode(rows, cols, mines, seed=seed_base + board))
        if len(candidates) >= target:
            break
    selected = balance_teacher_samples(candidates, max_states=max_states)
    if not selected:
        raise RuntimeError("teacher generation produced no usable states")
    return TeacherDataset(selected)
