"""Stable CLI entry point for the local M4-R3 dashboard."""

from __future__ import annotations

import argparse
import os
import threading
import webbrowser
from pathlib import Path

import uvicorn


def parse_args() -> argparse.Namespace:
    server_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--run-dir", default=str(server_root / "storage" / "runs" / "m4-r3"))
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--no-reload", action="store_true")
    parser.add_argument("--tag", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["MSAI_RUN_DIR"] = str(Path(args.run_dir).resolve())
    url = f"http://{args.host}:{args.port}"
    print(f"[dashboard] local R3 dashboard: {url}", flush=True)
    if not args.no_open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    options = {
        "host": args.host,
        "port": args.port,
        "log_level": "warning",
        "reload": not args.no_reload,
    }
    if not args.no_reload:
        options["reload_dirs"] = [str(Path(__file__).parent)]
    uvicorn.run("trainer.dashboard:app", **options)


if __name__ == "__main__":
    main()
else:
    from trainer.dashboard_app import app

    __all__ = ["app"]

