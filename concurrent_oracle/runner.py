"""Concurrent harness runner with subprocess isolation."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .observation import observation_from_harness, save_observation
from .process_manager import ProcessManager, signal_name
from .schemas import (
    ConcurrentHarnessArtifact,
    ConcurrentTestPlan,
    ExecutionObservation,
    ExecutionStatus,
    write_json,
)

ASAN_SO = "/root/tf_fuzz/lib/python3.10/site-packages/asan_with_fuzzer.so"


class ConcurrentRunner:
    def __init__(
        self,
        output_dir: str | Path,
        python_path: Optional[str] = None,
        preload_path: Optional[str] = ASAN_SO,
        timeout_grace: int = 30,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.python_path = python_path or sys.executable
        self.timeout_grace = timeout_grace
        self.pm = ProcessManager(self.python_path, preload_path=preload_path)

    def run(self, artifact: ConcurrentHarnessArtifact, plan: ConcurrentTestPlan) -> ExecutionObservation:
        harness = Path(artifact.concurrent_harness).resolve()
        config_id = str(plan.runtime.get("config_id") or "config")
        exec_dir = self.output_dir / "executions" / plan.plan_id / config_id
        exec_dir.mkdir(parents=True, exist_ok=True)
        obs_path = exec_dir / "observation.json"
        stdout_path = exec_dir / "stdout.log"
        stderr_path = exec_dir / "stderr.log"
        cfg_path = exec_dir / "case_config.json"
        cases_path = self.output_dir / "captured_cases" / plan.plan_id / "cases.json"

        cfg = {
            "plan_id": plan.plan_id,
            "config_id": config_id,
            "runtime": plan.runtime,
            "cases_path": str(cases_path),
            "output_path": str(obs_path),
        }
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

        pr = self.pm.run(
            [self.python_path, str(harness), str(cfg_path)],
            cwd=exec_dir,
            timeout=plan.timeout_seconds + self.timeout_grace,
        )
        stdout_path.write_text(pr.stdout or "", encoding="utf-8")
        stderr_path.write_text(pr.stderr or "", encoding="utf-8")

        raw: Optional[Dict[str, Any]] = None
        if obs_path.exists():
            try:
                raw = json.loads(obs_path.read_text(encoding="utf-8"))
            except Exception:
                raw = None

        if pr.timed_out:
            obs = ExecutionObservation(
                plan_id=plan.plan_id,
                config_id=config_id,
                execution_status=ExecutionStatus.TIMEOUT,
                process_exit_code=None,
                process_signal="SIGKILL",
                observation_complete=False,
                tasks=[],
                task_progress={"submitted": int(plan.runtime.get("external_threads", 0)),
                               "started": 0, "completed": 0,
                               "failed": 0, "pending": 0},
                runtime={},
                diagnostics={"stdout": str(stdout_path), "stderr": str(stderr_path),
                             "duration": pr.duration},
                harness_path=str(harness),
                config=cfg,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        elif raw is not None:
            obs = observation_from_harness(
                raw,
                plan_id=plan.plan_id,
                config_id=config_id,
                harness_path=str(harness),
                config=cfg,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                process_exit_code=pr.rc,
                process_signal=pr.signal,
            )
            if pr.rc not in (0, None) and obs.execution_status == ExecutionStatus.COMPLETED:
                obs.execution_status = ExecutionStatus.PARTIAL
        else:
            status = ExecutionStatus.CRASH if (pr.rc is not None and pr.rc < 0) else ExecutionStatus.SETUP_ERROR
            obs = ExecutionObservation(
                plan_id=plan.plan_id,
                config_id=config_id,
                execution_status=status,
                process_exit_code=pr.rc,
                process_signal=pr.signal or signal_name(pr.rc),
                observation_complete=False,
                tasks=[],
                task_progress={},
                runtime={},
                diagnostics={"stdout": str(stdout_path), "stderr": str(stderr_path),
                             "duration": pr.duration},
                harness_path=str(harness),
                config=cfg,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        save_observation(obs, exec_dir / "execution.json")
        return obs
