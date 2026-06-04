"""Solver soundness gate (M2 parity for the Python port): false-positive-0.

Drives many real (mine-bearing) boards with the solver and asserts that every cell the
solver claims SAFE is not a mine, and every cell it claims MINE is a mine. A single false
positive fails the suite (CLAUDE.md §5). Also reports the no-guess logic-solve rate.
"""

from __future__ import annotations

from trainer.env import MinesweeperEnv
from trainer.solver import analyze, generate_no_guess_board, view_from_arrays


def _view(env: MinesweeperEnv):
    return view_from_arrays(env.rows, env.cols, env.mines, env.revealed, env.adjacent, env.flagged)


def _drive(env: MinesweeperEnv) -> tuple[bool, bool]:
    """Solver-only playthrough. Returns (solved_without_guessing, false_positive)."""
    center = (env.rows // 2) * env.cols + (env.cols // 2)
    env.step(center)  # first move bootstraps (env places mines safe-area on first reveal)
    while env.status in ("ready", "playing"):
        safe, mines = analyze(_view(env))
        for m in mines:  # claimed mine must really be a mine
            if env.mine_layout[m] != 1:
                return (False, True)
        for s in safe:  # claimed safe must NOT be a mine (the critical invariant)
            if env.mine_layout[s] != 0:
                return (False, True)
        if not safe:
            return (False, False)  # would need a guess
        for s in safe:
            if env.status not in ("ready", "playing") or env.revealed[s]:
                continue
            env.step(s)
    return (env.status == "won", False)


def _run(rows, cols, mines, n, seed0):
    fp = 0
    solved = 0
    for g in range(n):
        env = MinesweeperEnv(rows, cols, mines, seed=seed0 + g, first_click_policy="safe-area")
        env.reset()
        ok, false_pos = _drive(env)
        fp += 1 if false_pos else 0
        solved += 1 if ok else 0
    return fp, solved


def test_solver_no_false_positive_beginner():
    fp, solved = _run(9, 9, 10, 400, 1_000)
    assert fp == 0, f"{fp} false positives on beginner"
    assert solved > 0  # sanity: some boards are fully logic-solvable


def test_solver_no_false_positive_intermediate():
    fp, _ = _run(16, 16, 40, 120, 2_000)
    assert fp == 0, f"{fp} false positives on intermediate"


def test_solver_no_false_positive_expert():
    fp, _ = _run(16, 30, 99, 60, 3_000)
    assert fp == 0, f"{fp} false positives on expert"


def test_single_point_safe_neighbors():
    """A revealed 0 makes all neighbors safe; a satisfied number makes the rest safe."""
    env = MinesweeperEnv(9, 9, 10, seed=42, first_click_policy="safe-area")
    env.reset()
    env.step(40)  # center; cascades from a 0
    safe, mines = analyze(_view(env))
    for s in safe:
        assert env.mine_layout[s] == 0
    for m in mines:
        assert env.mine_layout[m] == 1


def test_no_guess_generation_clears_without_guessing():
    """An NG board always has a certain-safe cell each round → batch reveal clears it."""
    for r, c, m in [(9, 9, 10), (16, 16, 40)]:
        env = generate_no_guess_board(r, c, m, seed_base=50_000)
        assert env is not None, f"no NG board found for {r}x{c}/{m}"
        guard = 0
        while env.status in ("ready", "playing") and guard < env.n + 5:
            guard += 1
            safe, _ = analyze(_view(env))
            assert safe, "an NG board must always expose a certain-safe cell"
            for s in safe:
                if env.status in ("ready", "playing") and not env.revealed[s]:
                    if env.mine_layout[s] != 0:
                        raise AssertionError("solver claimed a mine as safe")
                    env.step(s)
        assert env.status == "won"


if __name__ == "__main__":
    for diff, (r, c, m), n, s in [
        ("beginner", (9, 9, 10), 400, 1_000),
        ("intermediate", (16, 16, 40), 120, 2_000),
        ("expert", (16, 30, 99), 60, 3_000),
    ]:
        fp, solved = _run(r, c, m, n, s)
        print(f"{diff}: false_positives={fp}/{n}  logic_solved={solved}/{n}", flush=True)
