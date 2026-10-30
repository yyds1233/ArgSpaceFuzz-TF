"""Minimal progress watchdog used by the runner/observation layer."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class WatchdogState:
    started_at: float
    last_progress_at: float
    process_timeout: float
    no_progress_timeout: float


class Watchdog:
    def __init__(self, process_timeout_seconds: float = 60, no_progress_timeout_seconds: float = 20):
        now = time.time()
        self.state = WatchdogState(
            started_at=now,
            last_progress_at=now,
            process_timeout=float(process_timeout_seconds),
            no_progress_timeout=float(no_progress_timeout_seconds),
        )

    def beat(self) -> None:
        self.state.last_progress_at = time.time()

    def process_expired(self) -> bool:
        return (time.time() - self.state.started_at) > self.state.process_timeout

    def no_progress_expired(self) -> bool:
        return (time.time() - self.state.last_progress_at) > self.state.no_progress_timeout

    def check(self) -> Optional[str]:
        if self.process_expired():
            return "PROCESS_TIMEOUT"
        if self.no_progress_expired():
            return "NO_PROGRESS"
        return None
