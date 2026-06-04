"""Logic solver — a faithful Python port of `@msai/core/solver` (M2, false-positive-0).

Mirrors the verified TypeScript solver so the dashboard/online play can run a solver-first
hybrid policy (solver makes every CERTAIN move; the model/probability only guesses). The
deduction pipeline is single-point → subset → endgame → CSP frontier enumeration, returning
only cells that are provably safe or provably mines (never a false positive — CLAUDE.md §5).

The solver reasons from PLAYER-VISIBLE info only (revealed numbers + covered mask); it never
sees the mine layout. `analyze` returns certain safe/mine sets; `solve_step` adds a
lowest-risk guess via exact combinatorial probabilities for when no certainty remains.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COMPONENT_CAP = 20  # max frontier-component size to enumerate (matches the TS solver)


@dataclass
class SolverView:
    """Player-visible board projection. `number` is -1 for covered cells."""

    rows: int
    cols: int
    mines: int
    revealed: np.ndarray  # bool, length n
    number: np.ndarray  # int8, adjacent count for revealed cells, -1 for covered
    flagged: np.ndarray  # bool, length n (treated as known mines)


def _neighbors(i: int, rows: int, cols: int) -> list[int]:
    r, c = divmod(i, cols)
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                out.append(nr * cols + nc)
    return out


def _build_constraints(view: SolverView, mine: np.ndarray, safe: np.ndarray):
    """One constraint per revealed number with undetermined neighbors: (cells, sum)."""
    rows, cols, n = view.rows, view.cols, view.rows * view.cols
    out = []
    for c in range(n):
        if not view.revealed[c]:
            continue
        k = int(view.number[c])
        if k < 0:
            continue
        known_mines = 0
        vars_: list[int] = []
        for nb in _neighbors(c, rows, cols):
            if view.revealed[nb]:
                continue
            if mine[nb]:
                known_mines += 1
            elif not safe[nb]:
                vars_.append(nb)
        if vars_:
            out.append((vars_, k - known_mines))
    return out


def _single_point(view: SolverView, mine: np.ndarray, safe: np.ndarray) -> bool:
    changed = False
    for cells, s in _build_constraints(view, mine, safe):
        if s == 0:
            for cell in cells:
                if not safe[cell]:
                    safe[cell] = True
                    changed = True
        elif s == len(cells):
            for cell in cells:
                if not mine[cell]:
                    mine[cell] = True
                    changed = True
    return changed


def _subset_rule(view: SolverView, mine: np.ndarray, safe: np.ndarray) -> bool:
    cons = _build_constraints(view, mine, safe)
    sets = [set(c) for c, _ in cons]
    by_cell: dict[int, list[int]] = {}
    for ci, (cells, _) in enumerate(cons):
        for cell in cells:
            by_cell.setdefault(cell, []).append(ci)

    changed = False
    for i, (a_cells, a_sum) in enumerate(cons):
        a_set = sets[i]
        considered: set[int] = set()
        for cell in a_cells:
            for cj in by_cell[cell]:
                if cj == i or cj in considered:
                    continue
                considered.add(cj)
                b_cells, b_sum = cons[cj]
                if len(b_cells) <= len(a_cells):  # want a strict superset
                    continue
                b_set = sets[cj]
                if not a_set.issubset(b_set):
                    continue
                diff = [x for x in b_cells if x not in a_set]
                if not diff:
                    continue
                d_remaining = b_sum - a_sum
                if d_remaining == 0:
                    for c in diff:
                        if not safe[c]:
                            safe[c] = True
                            changed = True
                elif d_remaining == len(diff):
                    for c in diff:
                        if not mine[c]:
                            mine[c] = True
                            changed = True
    return changed


def _endgame(view: SolverView, mine: np.ndarray, safe: np.ndarray) -> bool:
    n = view.rows * view.cols
    known_mines = 0
    undetermined: list[int] = []
    for i in range(n):
        if view.revealed[i]:
            continue
        if mine[i]:
            known_mines += 1
        elif not safe[i]:
            undetermined.append(i)
    remaining = view.mines - known_mines
    changed = False
    if remaining == 0:
        for u in undetermined:
            if not safe[u]:
                safe[u] = True
                changed = True
    elif undetermined and remaining == len(undetermined):
        for u in undetermined:
            if not mine[u]:
                mine[u] = True
                changed = True
    return changed


def _connected_components(cons):
    """Group constraints into components by shared cells (union-find)."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cells, _ in cons:
        for c in cells:
            parent.setdefault(c, c)
        for i in range(1, len(cells)):
            union(cells[0], cells[i])

    cells_by_root: dict[int, list[int]] = {}
    for c in list(parent.keys()):
        cells_by_root.setdefault(find(c), []).append(c)
    cons_by_root: dict[int, list] = {}
    for cells, s in cons:
        cons_by_root.setdefault(find(cells[0]), []).append((cells, s))
    return [(cells, cons_by_root.get(r, [])) for r, cells in cells_by_root.items()]


def _enumerate_component(cells: list[int], cons, cap: int = COMPONENT_CAP):
    """Exhaustively enumerate valid assignments via pruned backtracking. None if > cap."""
    V = len(cells)
    if V > cap:
        return None
    local = {c: i for i, c in enumerate(cells)}
    con_vars = [[local[c] for c in cc] for cc, _ in cons]
    con_sum = [s for _, s in cons]
    in_cons: list[list[int]] = [[] for _ in range(V)]
    for ci, vs in enumerate(con_vars):
        for v in vs:
            in_cons[v].append(ci)

    assign = [0] * V
    con_mines = [0] * len(cons)
    con_unassigned = [len(vs) for vs in con_vars]

    total = 0
    mine_count_by_var = [0] * V
    ways_by_mines = [0] * (V + 1)
    mine_ways_by_var_mines = [[0] * (V + 1) for _ in range(V)]

    def dfs(v: int, mines: int) -> None:
        nonlocal total
        if v == V:
            total += 1
            ways_by_mines[mines] += 1
            for i in range(V):
                if assign[i] == 1:
                    mine_count_by_var[i] += 1
                    mine_ways_by_var_mines[i][mines] += 1
            return
        affected = in_cons[v]
        for val in (0, 1):
            assign[v] = val
            for ci in affected:
                con_mines[ci] += val
                con_unassigned[ci] -= 1
            ok = True
            for ci in affected:
                if con_mines[ci] > con_sum[ci] or con_mines[ci] + con_unassigned[ci] < con_sum[ci]:
                    ok = False
                    break
            if ok:
                dfs(v + 1, mines + val)
            for ci in affected:
                con_mines[ci] -= val
                con_unassigned[ci] += 1
        assign[v] = 0

    dfs(0, 0)
    return {
        "cells": cells,
        "total": total,
        "mineCountByVar": mine_count_by_var,
        "waysByMines": ways_by_mines,
        "mineWaysByVarMines": mine_ways_by_var_mines,
    }


def _csp_certain(view: SolverView, mine: np.ndarray, safe: np.ndarray, cap: int = COMPONENT_CAP):
    cons = _build_constraints(view, mine, safe)
    if not cons:
        return False
    comps = _connected_components(cons)
    changed = False
    frontier_min = 0
    frontier_max = 0
    in_frontier: set[int] = set()

    for comp_cells, comp_cons in comps:
        e = _enumerate_component(comp_cells, comp_cons, cap)
        for c in comp_cells:
            in_frontier.add(c)
        if not e or e["total"] == 0:
            frontier_max += len(comp_cells)  # unenumerable — assume widest range (stay sound)
            continue
        for i, cell in enumerate(e["cells"]):
            mc = e["mineCountByVar"][i]
            if mc == 0:
                if not safe[cell]:
                    safe[cell] = True
                    changed = True
            elif mc == e["total"]:
                if not mine[cell]:
                    mine[cell] = True
                    changed = True
        mn, mx = None, 0
        for m, w in enumerate(e["waysByMines"]):
            if w > 0:
                mn = m if mn is None else min(mn, m)
                mx = max(mx, m)
        frontier_min += 0 if mn is None else mn
        frontier_max += mx

    # Sea = covered, undetermined cells not adjacent to any number. Only mines KNOWN OUTSIDE
    # the frontier count here (frontier mines are in frontier_min/max) — counting them twice
    # would make the bound unsound (the M2 false-positive bug fix).
    n = view.rows * view.cols
    known_outside = 0
    sea: list[int] = []
    for i in range(n):
        if view.revealed[i] or i in in_frontier:
            continue
        if mine[i]:
            known_outside += 1
        elif not safe[i]:
            sea.append(i)
    if sea:
        remaining = view.mines - known_outside
        sea_max = remaining - frontier_min
        sea_min = remaining - frontier_max
        if sea_max <= 0:
            for c in sea:
                if not safe[c]:
                    safe[c] = True
                    changed = True
        elif sea_min >= len(sea):
            for c in sea:
                if not mine[c]:
                    mine[c] = True
                    changed = True
    return changed


def _run_deductions(view: SolverView):
    n = view.rows * view.cols
    mine = np.zeros(n, dtype=bool)
    safe = np.zeros(n, dtype=bool)
    mine[view.flagged.astype(bool)] = True
    while True:
        a = _single_point(view, mine, safe)
        b = _subset_rule(view, mine, safe)
        c = _endgame(view, mine, safe)
        if a or b or c:
            continue
        if not _csp_certain(view, mine, safe):
            break
    return mine, safe


def analyze(view: SolverView):
    """Return (safe_cells, mine_cells): all CERTAIN covered cells (false-positive-0)."""
    mine, safe = _run_deductions(view)
    n = view.rows * view.cols
    safe_cells = [i for i in range(n) if not view.revealed[i] and safe[i]]
    mine_cells = [i for i in range(n) if not view.revealed[i] and mine[i]]
    return safe_cells, mine_cells


def _binomial(n: int, k: int) -> float:
    if k < 0 or k > n:
        return 0.0
    kk = min(k, n - k)
    result = 1.0
    for i in range(kk):
        result = result * (n - i) / (i + 1)
    return result


def _convolve(a: list[float], b: list[float]) -> list[float]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            out[i + j] += ai * bj
    return out


def compute_probabilities(view: SolverView, mine: np.ndarray, safe: np.ndarray, cap=COMPONENT_CAP):
    """Exact per-cell mine probability for undetermined cells (for the guessing fallback)."""
    probs: dict[int, float] = {}
    cons = _build_constraints(view, mine, safe)
    comps = _connected_components(cons)

    enums = []
    for comp_cells, comp_cons in comps:
        e = _enumerate_component(comp_cells, comp_cons, cap)
        if e and e["total"] > 0:
            enums.append(e)
        else:  # too large — local per-constraint estimate
            for cc, s in comp_cons:
                p = s / len(cc) if cc else 0.0
                for c in cc:
                    probs.setdefault(c, max(0.0, min(1.0, p)))

    polys = [e["waysByMines"] for e in enums]
    n = view.rows * view.cols
    in_comp: set[int] = set()
    for e in enums:
        in_comp.update(e["cells"])
    known_mines = 0
    sea: list[int] = []
    for i in range(n):
        if view.revealed[i]:
            continue
        if mine[i]:
            known_mines += 1
        elif not (safe[i] or i in in_comp):
            sea.append(i)
    sea_size = len(sea)
    remaining = view.mines - known_mines

    sea_poly = [_binomial(sea_size, s) for s in range(min(sea_size, remaining) + 1)]
    if not sea_poly:
        sea_poly = [1.0]

    all_comp = [1.0]
    for p in polys:
        all_comp = _convolve(all_comp, p)
    full = _convolve(all_comp, sea_poly)
    W = full[remaining] if 0 <= remaining < len(full) else 0.0

    for ci, e in enumerate(enums):
        rest = [1.0]
        for cj, p in enumerate(polys):
            if cj != ci:
                rest = _convolve(rest, p)
        rest = _convolve(rest, sea_poly)
        for vi, cell in enumerate(e["cells"]):
            mine_ways = e["mineWaysByVarMines"][vi]
            num = 0.0
            for m, w in enumerate(mine_ways):
                if w == 0:
                    continue
                r = remaining - m
                if 0 <= r < len(rest):
                    num += w * rest[r]
            if W > 0:
                probs[cell] = num / W
            else:
                probs[cell] = e["mineCountByVar"][vi] / e["total"] if e["total"] > 0 else 0.0

    if sea_size > 0:
        expected = 0.0
        for s, sp in enumerate(sea_poly):
            a = remaining - s
            if 0 <= a < len(all_comp):
                expected += s * all_comp[a] * sp
        if W > 0:
            p_sea = expected / W / sea_size
        else:
            p_sea = remaining / (sea_size + len(in_comp)) if remaining > 0 else 0.0
        for c in sea:
            probs[c] = p_sea
    return probs


def solve_step(view: SolverView) -> dict:
    """Certain safe/mine cells plus a lowest-risk guess when no safe cell is certain."""
    mine, safe = _run_deductions(view)
    n = view.rows * view.cols
    safe_cells = [i for i in range(n) if not view.revealed[i] and safe[i]]
    mine_cells = [i for i in range(n) if not view.revealed[i] and mine[i]]
    if safe_cells:
        return {"safe": safe_cells, "mines": mine_cells, "guess": None, "probabilities": None}
    probs = compute_probabilities(view, mine, safe)
    if not probs:
        return {"safe": safe_cells, "mines": mine_cells, "guess": None, "probabilities": None}
    best_cell, best_prob = min(probs.items(), key=lambda kv: kv[1])
    return {
        "safe": safe_cells,
        "mines": mine_cells,
        "guess": {"cell": best_cell, "mine_prob": best_prob},
        "probabilities": probs,
    }


def _solver_clears_env(env) -> bool:
    """Solver-only playthrough from env's current state; True if cleared without guessing."""
    while env.status in ("ready", "playing"):
        view = view_from_arrays(
            env.rows, env.cols, env.mines, env.revealed, env.adjacent, env.flagged
        )
        safe, _ = analyze(view)
        if not safe:
            return False
        for s in safe:
            if env.status not in ("ready", "playing") or env.revealed[s]:
                continue
            env.step(s)
    return env.status == "won"


def generate_no_guess_board(
    rows: int,
    cols: int,
    mines: int,
    seed_base: int,
    *,
    first_click: int | None = None,
    max_attempts: int = 800,
):
    """Reject-sample a board the solver can clear with NO guessing (an "NG" board).

    Returns a started MinesweeperEnv (first click revealed, status 'playing') ready to play,
    or None if none was found within max_attempts. The NG property is tied to the first
    click (mines are placed safe-area around it), so the caller must continue from this env.
    """
    from trainer.env import MinesweeperEnv

    fc = first_click if first_click is not None else (rows // 2) * cols + (cols // 2)
    for a in range(max_attempts):
        seed = seed_base + a
        chk = MinesweeperEnv(rows, cols, mines, seed=seed, first_click_policy="safe-area")
        chk.reset()
        chk.step(fc)
        if _solver_clears_env(chk):  # deterministic by (seed, fc) → rebuild a fresh start state
            play = MinesweeperEnv(rows, cols, mines, seed=seed, first_click_policy="safe-area")
            play.reset()
            play.step(fc)
            return play
    return None


def view_from_arrays(
    rows: int,
    cols: int,
    mines: int,
    revealed: np.ndarray,
    adjacent: np.ndarray,
    flagged: np.ndarray | None = None,
) -> SolverView:
    """Build a SolverView, masking adjacency to -1 for covered cells (no info leak)."""
    n = rows * cols
    rev = revealed.astype(bool)
    number = np.where(rev, adjacent.astype(np.int16), np.int16(-1)).astype(np.int16)
    flg = np.zeros(n, dtype=bool) if flagged is None else flagged.astype(bool)
    return SolverView(rows, cols, mines, rev, number, flg)


def view_from_encoding(state: np.ndarray, mines: int | None = None) -> SolverView:
    """Reconstruct a SolverView from the (11,H,W) encoding (no flags during model play).

    channel 0 = covered(1); channels 1..9 = revealed with adjacent count k; channel 10 =
    mines/(rows*cols). This lets the hybrid run from the same input the policy net sees.
    """
    c, h, w = state.shape
    n = h * w
    covered = state[0].reshape(n) > 0.5
    revealed = ~covered
    number = np.full(n, -1, dtype=np.int16)
    nums = state[1:10].reshape(9, n)
    rev_idx = np.flatnonzero(revealed)
    if rev_idx.size:
        number[rev_idx] = nums[:, rev_idx].argmax(axis=0).astype(np.int16)
    if mines is None:
        mines = int(round(float(state[10].reshape(-1)[0]) * n))
    return SolverView(h, w, mines, revealed, number, np.zeros(n, dtype=bool))
