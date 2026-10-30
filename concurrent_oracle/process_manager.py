"""Subprocess lifecycle management for concurrent harness runs."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ProcessResult:
    rc: Optional[int]
    signal: Optional[str]
    timed_out: bool
    stdout: str
    stderr: str
    duration: float


def signal_name(rc: Optional[int]) -> Optional[str]:
    if rc is None or rc >= 0:
        return None
    try:
        return signal.Signals(-rc).name
    except Exception:
        return f"SIG{-rc}"


class ProcessManager:
    def __init__(self, python_path: str, preload_path: Optional[str] = None):
        self.python_path = python_path
        self.preload_path = preload_path

    def build_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = os.environ.copy()
        env.pop("LD_PRELOAD", None)
        if self.preload_path and Path(self.preload_path).exists():
            env["LD_PRELOAD"] = self.preload_path
            env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        if extra:
            env.update(extra)
        return env

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float,
        env_extra: Optional[Dict[str, str]] = None,
    ) -> ProcessResult:
        env = self.build_env(env_extra)
        t0 = time.time()
        try:
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return ProcessResult(
                rc=p.returncode,
                signal=signal_name(p.returncode),
                timed_out=False,
                stdout=p.stdout or "",
                stderr=p.stderr or "",
                duration=time.time() - t0,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return ProcessResult(
                rc=None,
                signal="SIGKILL",
                timed_out=True,
                stdout=stdout or "",
                stderr=stderr or "",
                duration=time.time() - t0,
            )
