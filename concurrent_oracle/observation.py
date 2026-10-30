"""Normalize raw harness observations into ExecutionObservation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    ExecutionObservation,
    ExecutionStatus,
    TaskObservation,
)


def _status(value: Any, default: ExecutionStatus = ExecutionStatus.PARTIAL) -> ExecutionStatus:
    try:
        return ExecutionStatus(str(value))
    except Exception:
        return default


def observation_from_harness(
    raw: Dict[str, Any],
    *,
    plan_id: str,
    config_id: str,
    harness_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    stdout_path: Optional[str] = None,
    stderr_path: Optional[str] = None,
    process_exit_code: Optional[int] = None,
    process_signal: Optional[str] = None,
) -> ExecutionObservation:
    tasks: List[TaskObservation] = []
    for t in raw.get("tasks") or []:
        exceptions = t.get("exceptions") or []
        exc0 = exceptions[0] if exceptions else {}
        tasks.append(TaskObservation(
            task_id=int(t.get("task_id", -1)),
            input_id=str(t.get("input_id", "")),
            status=str(t.get("status", "UNKNOWN")),
            output=t.get("outputs"),
            exception_type=exc0.get("type") or t.get("exception_type"),
            exception_message=exc0.get("message") or t.get("exception_message"),
            traceback=exc0.get("traceback") or t.get("traceback"),
            started_at=t.get("started_at"),
            finished_at=t.get("finished_at"),
            metadata={
                "outputs": t.get("outputs"),
                "exceptions": exceptions,
                "trial_records": t.get("trial_records") or [],
                "trials": t.get("trials"),
                "complete": t.get("complete"),
                "trials_expected": t.get("trials_expected"),
                "trials_completed": t.get("trials_completed"),
                "trials_failed": t.get("trials_failed"),
                "trial_status_counts": t.get("trial_status_counts"),
                "expected_calls": t.get("expected_calls"),
                "actual_calls": t.get("actual_calls"),
                "arg_conversions": t.get("arg_conversions") or [],
            },
        ))
    return ExecutionObservation(
        plan_id=plan_id,
        config_id=config_id,
        run_id=raw.get("run_id"),
        mode=raw.get("mode"),
        trials=raw.get("trials"),
        execution_status=_status(raw.get("execution_status"), ExecutionStatus.PARTIAL),
        process_exit_code=process_exit_code,
        process_signal=process_signal,
        observation_complete=bool(raw.get("observation_complete", False)),
        tasks=tasks,
        task_progress=dict(raw.get("task_progress") or {}),
        runtime=dict(raw.get("runtime") or {}),
        diagnostics={
            "stdout": stdout_path,
            "stderr": stderr_path,
            "raw_error": raw.get("error"),
            "raw_traceback": raw.get("traceback"),
            "shared_input_id": raw.get("shared_input_id"),
        },
        harness_path=harness_path,
        config=dict(config or raw.get("config") or {}),
        baseline=raw.get("baseline") or {},
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def save_observation(observation: ExecutionObservation, path: str | Path) -> Path:
    from .schemas import write_json
    return write_json(path, observation)
