"""Small dependency-free rotating stream for detached runtime logs."""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from pathlib import Path
from typing import TextIO


class RotatingTextStream:
    def __init__(
        self,
        path: str | Path,
        *,
        max_bytes: int,
        backups: int,
        mirror: TextIO | None = None,
    ) -> None:
        if max_bytes < 1 or backups < 0:
            raise ValueError("invalid log rotation limits")
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = backups
        self.mirror = mirror
        self.encoding = "utf-8"
        self.errors = "replace"
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding=self.encoding, buffering=1)

    def _rollover(self) -> None:
        self._handle.close()
        try:
            if self.backups:
                oldest = self.path.with_name(f"{self.path.name}.{self.backups}")
                oldest.unlink(missing_ok=True)
                for index in range(self.backups - 1, 0, -1):
                    source = self.path.with_name(f"{self.path.name}.{index}")
                    if source.exists():
                        os.replace(
                            source,
                            self.path.with_name(f"{self.path.name}.{index + 1}"),
                        )
                if self.path.exists():
                    os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
            else:
                self.path.unlink(missing_ok=True)
        except OSError:
            # The dashboard has one runtime writer. Truncation is the safe fallback
            # if an antivirus scanner temporarily prevents a Windows rename.
            self.path.write_bytes(b"")
        self._handle = self.path.open("a", encoding=self.encoding, buffering=1)

    def write(self, value: str) -> int:
        if not value:
            return 0
        encoded_size = len(value.encode(self.encoding, errors=self.errors))
        with self._lock:
            current_size = self._handle.tell()
            if current_size and current_size + encoded_size > self.max_bytes:
                self._rollover()
            self._handle.write(value)
            self._handle.flush()
            if self.mirror is not None:
                self.mirror.write(value)
                self.mirror.flush()
        return len(value)

    def flush(self) -> None:
        with self._lock:
            if self._handle.closed:
                return
            self._handle.flush()
            if self.mirror is not None:
                self.mirror.flush()

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()

    def isatty(self) -> bool:
        return bool(self.mirror and self.mirror.isatty())

    def writable(self) -> bool:
        return True


def install_rotating_stderr(
    path: str | Path,
    *,
    max_bytes: int = 10 * 1024 * 1024,
    backups: int = 2,
) -> RotatingTextStream:
    """Capture app-process stderr while retaining console output for interactive runs."""
    if isinstance(sys.stderr, RotatingTextStream):
        return sys.stderr
    original = sys.stderr
    mirror = original if original.isatty() else None
    stream = RotatingTextStream(
        path,
        max_bytes=max_bytes,
        backups=backups,
        mirror=mirror,
    )
    sys.stderr = stream  # type: ignore[assignment]

    # Uvicorn configures handlers before importing the ASGI app. Redirect handlers
    # that captured the previous stderr so they obey the same size bound.
    loggers = [logging.getLogger()]
    loggers.extend(
        value
        for value in logging.root.manager.loggerDict.values()
        if isinstance(value, logging.Logger)
    )
    for logger in loggers:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream is original:
                handler.setStream(stream)
    atexit.register(stream.close)
    return stream
