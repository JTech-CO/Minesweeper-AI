"""Play a real minesweeper.online game in a server-side browser (Playwright).

The dashboard's model decides moves; Playwright reads the live board from the DOM and
clicks cells on the real page, streaming each move to the dashboard. **Personal demo /
bench use only**, with human-like click delays (rate limiting). Do not submit automated
games to public/ranked leaderboards; respect the site's terms of service.

DOM contract (minesweeper.online) is isolated in `DOM` + the JS snippets below, so it can
be corrected from DevTools without touching the play logic (run-book 10). Verified against
the live site (a real /start/1 game) on 2026-06-05: cells are 0-indexed and the board is
drawn ~2s after DOMContentLoaded, so run() waits for the cells before reading.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable

import numpy as np

from trainer.encoding import NUM_CHANNELS, encode

# --- DOM contract (minesweeper.online) -----------------------------------------------------
# Cells are <div id="cell_{col}_{row}"> (0-indexed: cell_0_0 .. cell_{cols-1}_{rows-1}),
# class "cell size24 ...", whose state class is:
#   hd_closed | hd_flag | hd_opened + hd_type{0..8} | hd_type1{0,1,2}=mine/exploded/wrong
JS_READ_BOARD = r"""
() => {
  const cells = document.querySelectorAll('[id^="cell_"]');
  if (!cells.length) return null;
  let maxC = 0, maxR = 0;
  const data = [];
  cells.forEach((el) => {
    const m = el.id.match(/^cell_(\d+)_(\d+)$/);
    if (!m) return;
    const c = +m[1], r = +m[2];
    if (c > maxC) maxC = c;
    if (r > maxR) maxR = r;
    const cls = el.className;
    let state = 'closed', num = 0;
    if (cls.includes('hd_flag')) state = 'flag';
    else if (cls.includes('hd_opened')) {
      state = 'open';
      for (let k = 0; k <= 8; k++) if (cls.includes('hd_type' + k)) { num = k; }
    } else if (/hd_type1[0-2]/.test(cls)) {
      state = (cls.includes('hd_type11')) ? 'exploded' : 'mine';
    }
    data.push({ c, r, state, num });
  });
  let mines = null;
  const mineEls = document.querySelectorAll('#top_area_mines [id^="top_area_mines_"]');
  if (mineEls.length) {
    let s = '';
    mineEls.forEach((e) => {
      const mm = (e.className.match(/hd_top-area-num(\d)/) || [])[1];
      if (mm) s += mm;
    });
    if (s) mines = parseInt(s, 10);
  }
  return { cols: maxC, rows: maxR, cells: data, mines };
}
"""

JS_FACE = r"""
() => {
  const f = document.querySelector('#top_area_face') || document.querySelector('.top-area-face');
  if (!f) return 'playing';
  const c = f.className;
  if (c.includes('-win')) return 'won';
  if (c.includes('-lose')) return 'lost';
  return 'playing';
}
"""

DOM = {"cell_id": "cell_{col}_{row}"}  # 0-indexed col,row (verified live)

_ERR_NO_PW = "playwright not installed; run: playwright install chromium"
_ERR_NO_BOARD = "no board at this URL (check the game URL / DOM contract)"
_LOGIN_URL = "https://minesweeper.online/"


class _BoardView:
    """Minimal object exposing exactly the attributes encoding.encode() reads."""

    def __init__(
        self, rows: int, cols: int, mines: int, revealed: np.ndarray, adjacent: np.ndarray
    ):
        self.rows = rows
        self.cols = cols
        self.n = rows * cols
        self.mines = mines
        self.revealed = revealed
        self.adjacent = adjacent


def _estimate_mines(rows: int, cols: int) -> int:
    n = rows * cols
    return {81: 10, 256: 40, 480: 99}.get(n, max(1, round(0.16 * n)))


def parse_board(raw: dict) -> dict:
    """Convert the JS-extracted board into revealed/adjacent/flag arrays + dims.

    minesweeper.online uses 0-indexed cell ids (cell_0_0 .. cell_{cols-1}_{rows-1}). We
    derive dims and a base offset from the actual min/max coordinates so the mapping is
    correct regardless of whether the site is 0- or 1-indexed (verified 0-indexed live).
    """
    cells = raw["cells"]
    cs = [cell["c"] for cell in cells]
    rs = [cell["r"] for cell in cells]
    col0, row0 = min(cs), min(rs)
    cols = max(cs) - col0 + 1
    rows = max(rs) - row0 + 1
    n = rows * cols
    revealed = np.zeros(n, dtype=np.uint8)
    flagged = np.zeros(n, dtype=np.uint8)
    adjacent = np.zeros(n, dtype=np.uint8)
    cell_codes = np.full(
        n, -1, dtype=np.int8
    )  # for rendering: -1 hidden,0..8,-2 mine,-3 exploded,-4 flag
    for cell in cells:
        i = (cell["r"] - row0) * cols + (cell["c"] - col0)
        if not (0 <= i < n):
            continue
        st = cell["state"]
        if st == "open":
            revealed[i] = 1
            adjacent[i] = cell["num"]
            cell_codes[i] = cell["num"]
        elif st == "flag":
            flagged[i] = 1
            cell_codes[i] = -4
        elif st == "exploded":
            cell_codes[i] = -3
        elif st == "mine":
            cell_codes[i] = -2
        else:
            cell_codes[i] = -1
    mines = raw.get("mines") or _estimate_mines(rows, cols)
    return {
        "rows": rows,
        "cols": cols,
        "col0": col0,
        "row0": row0,
        "mines": mines,
        "revealed": revealed,
        "flagged": flagged,
        "adjacent": adjacent,
        "cell_codes": cell_codes.tolist(),
    }


class OnlinePlayer:
    """Drives minesweeper.online games with the dashboard's model via Playwright.

    Plays continuously, auto-restarting a fresh game via the smiley face until stopped,
    and measures per-game time + clicks-per-second (CPS). With a persistent profile dir the
    browser stays logged in across runs (log in once via login()), so the site records the
    games natively. Personal/demo use, rate-limited (self.delay between clicks); do not
    submit automated games to public/ranked leaderboards — respect the site's ToS.
    """

    def __init__(
        self,
        play_act,
        *,
        delay: float = 0.3,
        headless: bool = False,
        profile_dir: str | None = None,
    ):
        self.play_act = play_act  # (state, mask) -> action index
        self.delay = max(0.0, delay)  # seconds between clicks (rate limit)
        self.headless = headless
        self.profile_dir = profile_dir  # persistent Chrome profile → keeps login session
        # threading.Event (not asyncio): the dashboard runs the player on a dedicated loop
        # in a worker thread, so stop is signalled from a different thread than run()'s loop.
        self._stop = threading.Event()
        self.status = "idle"

    def request_stop(self) -> None:
        self._stop.set()

    async def _open(self, pw):
        """Open (closable, page). Persistent context keeps the login session if a dir is set."""
        if self.profile_dir:
            ctx = await pw.chromium.launch_persistent_context(
                self.profile_dir, headless=self.headless
            )
            page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            return ctx, page
        browser = await pw.chromium.launch(headless=self.headless)
        return browser, await browser.new_page()

    async def login(self, emit: Callable[[dict], None]) -> None:
        """Open the site so the user logs in once; the persistent profile keeps the session."""
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            emit({"type": "online_error", "msg": _ERR_NO_PW})
            return
        if not self.profile_dir:
            emit({"type": "online_error", "msg": "login requires a persistent profile dir"})
            return
        self._stop.clear()
        self.status = "login"
        emit({"type": "online_status", "status": "login", "url": _LOGIN_URL})
        async with async_playwright() as pw:
            closable, page = await self._open(pw)
            try:
                await page.goto(_LOGIN_URL, wait_until="domcontentloaded", timeout=30000)
                emit({"type": "online_login_wait"})
                while not self._stop.is_set():  # stay open until the user clicks stop
                    await asyncio.sleep(0.5)
            except Exception as e:  # noqa: BLE001 - surface to the dashboard
                emit({"type": "online_error", "msg": f"{type(e).__name__}: {e}"})
            finally:
                await closable.close()  # flush cookies/session to the profile dir
                self._stop.clear()
                self.status = "idle"
                emit({"type": "online_status", "status": "idle", "url": _LOGIN_URL})

    async def run(self, url: str, emit: Callable[[dict], None]) -> None:
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            emit({"type": "online_error", "msg": _ERR_NO_PW})
            return

        self._stop.clear()
        self.status = "launching"
        emit({"type": "online_status", "status": self.status, "url": url})
        games = wins = losses = 0
        best_ms: float | None = None
        async with async_playwright() as pw:
            closable, page = await self._open(pw)
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                self.status = "playing"
                first = True
                while not self._stop.is_set():
                    if not first:  # restart a fresh game via the smiley face
                        try:
                            await page.click("#top_area_face", timeout=5000)
                        except Exception:  # noqa: BLE001 - fall back to reloading the URL
                            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    first = False
                    try:
                        await page.wait_for_selector('[id^="cell_"]', timeout=20000)
                        await asyncio.sleep(0.3)  # let the grid finish painting
                    except Exception:  # noqa: BLE001
                        emit({"type": "online_error", "msg": _ERR_NO_BOARD})
                        break
                    result = await self._play_one_game(page, emit, games)
                    if result is None:  # stopped mid-game (or no board)
                        break
                    won, moves, ms = result
                    games += 1
                    if won:
                        wins += 1
                        if best_ms is None or ms < best_ms:
                            best_ms = ms
                    else:
                        losses += 1
                    cps = moves / (ms / 1000) if ms > 0 else 0.0
                    emit(
                        {
                            "type": "online_stats",
                            "games": games,
                            "wins": wins,
                            "losses": losses,
                            "lastWon": won,
                            "lastMoves": moves,
                            "lastMs": round(ms),
                            "lastCps": round(cps, 2),
                            "bestMs": round(best_ms) if best_ms is not None else None,
                        }
                    )
            except Exception as e:  # noqa: BLE001 - surface to the dashboard
                emit({"type": "online_error", "msg": f"{type(e).__name__}: {e}"})
            finally:
                await closable.close()
                self.status = "stopped"
                emit({"type": "online_status", "status": self.status, "url": url})

    async def _play_one_game(self, page, emit, game_index: int):
        """Play one game to completion. Returns (won, moves, elapsed_ms), or None if stopped.

        Timing starts at the first click (like the site's timer). CPS = moves / elapsed.
        """
        move = 0
        t0: float | None = None
        while not self._stop.is_set():
            raw = await page.evaluate(JS_READ_BOARD)
            if not raw or not raw.get("cells"):
                emit({"type": "online_error", "msg": _ERR_NO_BOARD})
                return None
            board = parse_board(raw)
            rows, cols = board["rows"], board["cols"]
            if move == 0:
                emit(
                    {
                        "type": "online_start",
                        "rows": rows,
                        "cols": cols,
                        "mines": board["mines"],
                        "url": page.url,
                        "game": game_index + 1,
                    }
                )
            elapsed_ms = 0.0 if t0 is None else (time.monotonic() - t0) * 1000
            cps = move / (elapsed_ms / 1000) if elapsed_ms > 0 else 0.0
            face = await page.evaluate(JS_FACE)
            emit(
                {
                    "type": "online_board",
                    "rows": rows,
                    "cols": cols,
                    "cells": board["cell_codes"],
                    "move": move,
                    "status": face,
                    "mines": board["mines"],
                    "elapsedMs": round(elapsed_ms),
                    "cps": round(cps, 2),
                    "game": game_index + 1,
                }
            )
            # A revealed mine/explosion is a definitive loss even if the face DOM drifts.
            exploded = -3 in board["cell_codes"] or -2 in board["cell_codes"]
            if face in ("won", "lost") or exploded:
                return (face == "won" and not exploded), move, elapsed_ms
            covered = (board["revealed"] == 0) & (board["flagged"] == 0)
            mask = covered.astype(np.uint8)
            if mask.sum() == 0:
                return (face == "won"), move, elapsed_ms
            view = _BoardView(rows, cols, board["mines"], board["revealed"], board["adjacent"])
            state = encode(view)
            if state.shape[0] != NUM_CHANNELS:
                return (face == "won"), move, elapsed_ms
            action = self.play_act(state, mask)
            c = board["col0"] + (action % cols)
            r = board["row0"] + (action // cols)
            await page.click(f"#cell_{c}_{r}", timeout=5000)
            if t0 is None:
                t0 = time.monotonic()  # start the clock at the first click
            move += 1
            await asyncio.sleep(self.delay)  # rate limit (human-like)
        return None  # stopped
