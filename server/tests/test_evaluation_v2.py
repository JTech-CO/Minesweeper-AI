from __future__ import annotations

import numpy as np
import pytest

from trainer.evaluation import evaluate_policy, get_suite, wilson_interval
from trainer.evaluation.suites import training_seed


class FirstLegalPolicy:
    def act(self, _state: np.ndarray, legal_mask: np.ndarray) -> int:
        return int(np.flatnonzero(legal_mask)[0])


def test_seed_suites_are_disjoint_and_bounded():
    validation = set(get_suite("validation", games=50).seeds)
    test = set(get_suite("test", games=50).seeds)
    assert validation.isdisjoint(test)
    assert training_seed(123, run_seed=4) not in validation | test
    with pytest.raises(ValueError):
        get_suite("test", games=5_001)


def test_wilson_interval_contains_observed_rate():
    low, high = wilson_interval(23, 100)
    assert low < 0.23 < high
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_evaluation_is_exactly_reproducible():
    seeds = get_suite("smoke", games=12).seeds
    first = evaluate_policy(
        FirstLegalPolicy(), 5, 5, 3, seeds=seeds, suite_name="smoke"
    )
    second = evaluate_policy(
        FirstLegalPolicy(), 5, 5, 3, seeds=seeds, suite_name="smoke"
    )
    assert first == second
    assert first.games == 12
    assert first.opening == "center"
