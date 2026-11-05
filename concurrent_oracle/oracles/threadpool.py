"""O8: basic task-level CPU ThreadPool progress check."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    ExecutionStatus,
    OracleEntry,
    OracleStatus,
)
from .base import BaseOracle


class ThreadpoolOracle(BaseOracle):
    oracle_id = "O8"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O8 not in plan")
        if self.contract_status(plan) == ContractStatus.UNSUPPORTED:
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE,
                               "contract marks this scenario unsupported")
        progress = dict(observation.task_progress or {})
        diag = observation.diagnostics or {}
        evidence = {
            "execution_status": observation.execution_status.value,
            "task_progress": progress,
            "process_repetitions": diag.get("process_repetitions", 1),
            "last_event": diag.get("last_event"),
            "no_progress": diag.get("no_progress"),
            "events": diag.get("events") or diag.get("test_events"),
        }
        if observation.execution_status == ExecutionStatus.TIMEOUT:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "execution timed out before all tasks completed; liveness candidate, "
                "not automatically a deadlock",
                evidence=evidence,
            )
        if observation.execution_status in (ExecutionStatus.CRASH, ExecutionStatus.SETUP_ERROR):
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"execution ended with {observation.execution_status.value}",
                evidence=evidence,
            )
        submitted = int(progress.get("submitted", len(observation.tasks)))
        completed = int(progress.get("completed", 0))
        failed = int(progress.get("failed", 0))
        pending = int(progress.get("pending", 0))
        if not observation.observation_complete or failed or pending or completed < submitted:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"incomplete task progress: submitted={submitted} completed={completed} "
                f"failed={failed} pending={pending}",
                evidence=evidence,
            )
        return OracleEntry(self.oracle_id, OracleStatus.PASS,
                           "all submitted tasks completed with no pending work",
                           evidence=evidence)
