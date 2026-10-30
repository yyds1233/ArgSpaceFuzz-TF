"""Subprocess lifecycle management for concurrent harness runs."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .watchdog import Watchdog


@dataclass
class ProcessResult:
    rc: Optional[int]
    signal: Optional[str]
    timed_out: bool
    stdout: str
    stderr: str
    duration: float
    no_progress: bool = False
    last_event: Optional[str] = None
    last_event_age: Optional[float] = None
    stack_signal_sent: bool = False


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

    @staticmethod
    def _read_last_event(path: str | Path, last_size: int):
        p = Path(path)
        try:
            size = p.stat().st_size
            if size == last_size:
                return None, last_size
            with p.open("rb") as f:
                f.seek(max(0, size - 8192))
                lines = f.read().splitlines()
            for line in reversed(lines):
                try:
                    return json.loads(line.decode("utf-8")), size
                except Exception:
                    continue
        except FileNotFoundError:
            return None, last_size
        except Exception:
            return None, last_size
        return None, last_size

    def run(
        self,
        cmd: list[str],
        *,
        cwd: str | Path,
        timeout: float,
        env_extra: Optional[Dict[str, str]] = None,
        events_path: Optional[str | Path] = None,
        no_progress_timeout: Optional[float] = None,
    ) -> ProcessResult:
        env = self.build_env(env_extra)
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        stop = threading.Event()
        state = {
            "no_progress": False,
            "last_event": None,
            "last_event_age": None,
            "stack_signal_sent": False,
        }

        def monitor():
            last_size = -1
            seen_event = False
            last_progress = time.monotonic()
            watchdog = Watchdog(timeout, no_progress_timeout or timeout)
            while not stop.wait(0.2):
                if proc.poll() is not None:
                    return
                if events_path:
                    rec, last_size = self._read_last_event(events_path, last_size)
                    if rec:
                        seen_event = True
                        state["last_event"] = rec.get("event")
                        try:
                            last_progress = float(rec.get("ts", time.monotonic()))
                        except Exception:
                            last_progress = time.monotonic()
                        state["last_event_ts"] = last_progress
                        watchdog.state.last_progress_at = last_progress
                # Do not enforce no-progress before the first event: TensorFlow
                # import/setup is covered by the overall process timeout, and a
                # cold import may legitimately exceed the no-progress budget.
                if not seen_event or no_progress_timeout is None:
                    continue
                age = time.monotonic() - last_progress
                if watchdog.no_progress_expired():
                    state["no_progress"] = True
                    state["last_event_age"] = age
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGUSR1)
                        state["stack_signal_sent"] = True
                    except Exception:
                        pass
                    time.sleep(0.5)
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    return

        monitor_thread = threading.Thread(target=monitor, name="co-watchdog", daemon=True)
        monitor_thread.start()
        timed_out = False
        try:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except Exception:
                    stdout, stderr = "", ""
        finally:
            stop.set()
            monitor_thread.join(timeout=1.0)
        if state["no_progress"]:
            timed_out = True
        if state.get("last_event_ts"):
            state["last_event_age"] = time.monotonic() - state["last_event_ts"]
        return ProcessResult(
            rc=proc.returncode,
            signal=signal_name(proc.returncode),
            timed_out=timed_out,
            stdout=stdout or "",
            stderr=stderr or "",
            duration=time.monotonic() - t0,
            no_progress=bool(state["no_progress"]),
            last_event=state["last_event"],
            last_event_age=state["last_event_age"],
            stack_signal_sent=bool(state["stack_signal_sent"]),
        )
