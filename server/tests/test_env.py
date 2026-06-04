"""M4 DoD #4: the Python env is deterministic and its rules match the TS engine
(packages/core/src/board.ts). Cases are ported from packages/core/test/board.test.ts
and env.test.ts to keep the dual implementation in sync (CLAUDE.md §5)."""

from __future__ import annotations

import pytest

from trainer.env import DIFFICULTIES, MinesweeperEnv, neighbors


def revealed_indices(env: MinesweeperEnv) -> list[int]:
    return [i for i in range(env.n) if env.revealed[i]]


def test_neighbors_clamp_on_3x3():
    assert sorted(neighbors(0, 3, 3)) == [1, 3, 4]
    assert sorted(neighbors(4, 3, 3)) == [0, 1, 2, 3, 5, 6, 7, 8]
    assert sorted(neighbors(8, 3, 3)) == [4, 5, 7]


def test_determinism_same_seed_same_first_click():
    def layout() -> list[int]:
        env = MinesweeperEnv(9, 9, 10, seed=2026)
        env.step(40)
        return env.mine_layout.tolist()

    assert layout() == layout()


def test_place_mines_count():
    env = MinesweeperEnv(16, 30, 99, seed=5)
    env.step(0)
    assert int(env.mine_layout.sum()) == 99


def test_safe_area_first_click_opens_cascade():
    click = 40  # center of 9x9
    for seed in range(200):
        env = MinesweeperEnv(9, 9, 10, seed=seed)
        env.step(click)
        assert env.mine_layout[click] == 0
        for nb in neighbors(click, 9, 9):
            assert env.mine_layout[nb] == 0
        assert env.adjacent[click] == 0  # guaranteed opening
        assert env.status != "lost"


def test_cascade_clears_board_win():
    # 3x3, mine at corner 0:  M 1 . / 1 1 . / . . .
    env = MinesweeperEnv(3, 3, 1)
    env.set_mines([0])
    _, reward, done, info = env.step(8)  # click a 0-cell
    assert done and info["won"]
    assert env.status == "won"
    assert revealed_indices(env) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert reward == env.reward_cfg.win


def test_split_openings_partial_then_win():
    # 1x5: 0 1 M 1 0 (mine at 2)
    env = MinesweeperEnv(1, 5, 1)
    env.set_mines([2])
    env.step(0)
    assert revealed_indices(env) == [0, 1]
    assert env.status == "playing"
    _, _, done, info = env.step(4)
    assert done and info["won"]
    assert revealed_indices(env) == [0, 1, 3, 4]


def test_hitting_mine_loses():
    env = MinesweeperEnv(3, 3, 1)
    env.set_mines([0])
    _, reward, done, info = env.step(0)
    assert done and info["hit_mine"]
    assert env.status == "lost"
    assert env.exploded_at == 0
    assert reward == env.reward_cfg.loss


def test_flag_blocks_cascade():
    env = MinesweeperEnv(3, 3, 1)
    env.set_mines([0])
    assert env.toggle_flag(1)
    env.step(8)
    assert env.status == "playing"  # flagged cell stays hidden → not all safe revealed
    assert env.revealed[1] == 0 and env.flagged[1] == 1
    assert revealed_indices(env) == [2, 3, 4, 5, 6, 7, 8]


def test_legal_action_mask_lifecycle():
    env = MinesweeperEnv(9, 9, 10, seed=7)
    assert env.legal_action_mask().sum() == 81  # ready → all legal
    env.step(40)
    mask = env.legal_action_mask()
    for i in range(81):
        if env.revealed[i]:
            assert mask[i] == 0
    assert mask[40] == 0


def test_terminal_and_range_errors():
    # 1x2 with 1 mine: first click is safe (mine forced to the other cell) → instant win.
    env = MinesweeperEnv(1, 2, 1)
    _, _, done, info = env.step(0)
    assert done and info["won"]
    assert env.legal_action_mask().sum() == 0
    with pytest.raises(RuntimeError):
        env.step(1)
    env2 = MinesweeperEnv(9, 9, 10)
    with pytest.raises(ValueError):
        env2.step(-1)
    with pytest.raises(ValueError):
        env2.step(81)


def test_presets_match_spec():
    assert DIFFICULTIES["beginner"] == (9, 9, 10)
    assert DIFFICULTIES["intermediate"] == (16, 16, 40)
    assert DIFFICULTIES["expert"] == (16, 30, 99)


def test_deterministic_action_sequence_reproduces():
    def run() -> tuple[list[int], str]:
        env = MinesweeperEnv(9, 9, 10, seed=123)
        env.reset()
        rewards: list[int] = []
        for a in (40, 0, 8, 72, 80, 4, 36):
            if env.status in ("won", "lost"):
                break
            _, r, _, _ = env.step(a)
            rewards.append(round(r, 3))
        return rewards, env.render()

    r1, board1 = run()
    r2, board2 = run()
    assert r1 == r2
    assert board1 == board2
