"""Observability & structured logging for HybAligner.

Provides a lightweight structured logger with JSON and human-readable
output modes. Designed for pipeline monitoring and debugging.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
from pathlib import Path


# ---------------------------------------------------------------------------
# Log entry
# ---------------------------------------------------------------------------
@dataclass
class LogEntry:
    """A single structured log entry."""
    event: str
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    elapsed_s: float = 0.0
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"event": self.event, "timestamp": self.timestamp, "elapsed_s": self.elapsed_s}
        d.update(self.data)
        return d


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
class Logger:
    """Structured logger for pipeline events."""

    def __init__(
        self,
        output: str = "stderr",
        format: str = "human",
        level: str = "info",
    ):
        self.format = format
        self.level = level
        self._entries: list = []
        self._start_time = time.perf_counter()
        self._file = None

        if output == "stderr":
            self._file = sys.stderr
        elif output == "stdout":
            self._file = sys.stdout
        elif output:
            self._file = open(output, 'a')

    def log(self, event: str, **kwargs):
        """Emit a structured log event."""
        elapsed = time.perf_counter() - self._start_time
        entry = LogEntry(event=event, elapsed_s=round(elapsed, 4), data=kwargs)
        self._entries.append(entry)

        if self._file:
            ts = time.strftime("%H:%M:%S")
            if self.format == "json":
                print(json.dumps(entry.to_dict()), file=self._file)
            else:
                # Human-readable: truncate data for readability
                kv = " ".join(f"{k}={v}" for k, v in kwargs.items())
                print(f"[{ts}] {event:30s} {kv}", file=self._file)
            self._file.flush()

    def get_entries(self) -> list:
        return [e.to_dict() for e in self._entries]

    def dump(self, path: str):
        """Write all log entries to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.get_entries(), f, indent=2)

    def close(self):
        if self._file and self._file not in (sys.stdout, sys.stderr):
            self._file.close()


# ---------------------------------------------------------------------------
# Global logger instance
# ---------------------------------------------------------------------------
_logger: Optional[Logger] = None


def init_logger(
    output: str = "stderr",
    format: str = "human",
    level: str = "info",
) -> Logger:
    """Initialize the global logger."""
    global _logger
    _logger = Logger(output=output, format=format, level=level)
    return _logger


def log(event: str, **kwargs):
    """Convenience function for logging.

    Usage:
        from obs.log import log
        log("alignment_done", read_id="read_001", score=42.5)
    """
    global _logger
    if _logger is None:
        _logger = Logger(output="stderr", format="human")
    _logger.log(event, **kwargs)


def get_logger() -> Logger:
    global _logger
    if _logger is None:
        _logger = Logger(output="stderr", format="human")
    return _logger
