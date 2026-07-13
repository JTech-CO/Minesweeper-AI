"""Immutable, non-overlapping seed suites.

Training seeds occupy the low integer range. Validation and test suites use separate
high ranges so checkpoint selection never observes the final test boards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeedSuite:
    name: str
    seed_base: int
    games: int

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_base, self.seed_base + self.games))


_SUITES = {
    "smoke": SeedSuite("smoke", 2_000_000_000, 32),
    "validation": SeedSuite("validation", 1_000_000_000, 1_000),
    "test": SeedSuite("test", 1_500_000_000, 5_000),
}


def get_suite(name: str, *, games: int | None = None) -> SeedSuite:
    try:
        suite = _SUITES[name]
    except KeyError as exc:
        raise ValueError(f"unknown evaluation suite {name!r}") from exc
    if games is None:
        return suite
    if games < 1 or games > suite.games:
        raise ValueError(f"games must be in 1..{suite.games} for {name}")
    return SeedSuite(suite.name, suite.seed_base, games)


def training_seed(episode: int, *, run_seed: int = 0) -> int:
    if episode < 0:
        raise ValueError("episode must be non-negative")
    seed = run_seed * 10_000_000 + episode
    if seed >= 900_000_000:
        raise ValueError("training seed range exhausted")
    return seed
