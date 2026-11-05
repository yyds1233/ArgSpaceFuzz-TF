"""O7: Python callback and generator lifecycle.

The callback recipe records enter/exit events, element IDs, propagated
exceptions and whether a new pipeline can be consumed after an expected
callback exception.  O7 deliberately does not require an exact callback count:
prefetching may execute more callbacks than the number of consumed elements.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

from ..schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    ExecutionStatus,
    OracleEntry,
    OracleStatus,
)
from .base import BaseOracle


class CallbackLifecycleOracle(BaseOracle):
    oracle_id = "O7"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O7 not in plan")
        oracle_contract = self.contract_status(plan)
        if oracle_contract == ContractStatus.UNSUPPORTED:
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE,
                               "contract marks this scenario unsupported")
        if observation.execution_status in (ExecutionStatus.TIMEOUT,
                                            ExecutionStatus.CRASH,
                                            ExecutionStatus.SETUP_ERROR):
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"execution ended with {observation.execution_status.value}",
                evidence={"execution_status": observation.execution_status.value},
            )
        if observation.execution_status != ExecutionStatus.COMPLETED:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               f"execution status is {observation.execution_status.value}")
        if not observation.observation_complete:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               "observation is incomplete")

        callbacks: List[Dict[str, Any]] = []
        for task in observation.tasks or []:
            record = (task.metadata or {}).get("callbacks")
            if isinstance(record, dict):
                callbacks.append({"task_id": task.task_id, **record})
            records = (task.metadata or {}).get("callback_records")
            if isinstance(records, list):
                for item in records:
                    if isinstance(item, dict):
                        callbacks.append({"task_id": task.task_id, **item})
        if not callbacks:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "no callback observation was recorded",
                evidence={"tasks": len(observation.tasks or [])},
            )

        problems: List[Dict[str, Any]] = []
        checked = 0
        for cb in callbacks:
            checked += 1
            entered = cb.get("entered_ids")
            exited = cb.get("exited_ids")
            expected_exception_at = cb.get("expected_exception_at")
            if isinstance(entered, list) and isinstance(exited, list):
                entered_c = Counter(map(str, entered))
                exited_c = Counter(map(str, exited))
                # The deliberately raising callback has an enter event but no
                # exit event; that one missing exit is expected.
                if expected_exception_at is not None:
                    key = str(expected_exception_at)
                    if entered_c.get(key):
                        entered_c[key] -= 1
                        if not entered_c[key]:
                            del entered_c[key]
                if entered_c != exited_c:
                    problems.append({
                        "kind": "enter_exit_mismatch",
                        "task_id": cb.get("task_id"),
                        "entered": entered[:20], "exited": exited[:20],
                    })
            if expected_exception_at is not None:
                if cb.get("exception_propagated") is False:
                    problems.append({
                        "kind": "exception_not_propagated",
                        "task_id": cb.get("task_id"),
                        "expected_at": expected_exception_at,
                    })
            if cb.get("subsequent_pipeline_ok") is False:
                problems.append({
                    "kind": "post_exception_pipeline_failed",
                    "task_id": cb.get("task_id"),
                })
            if cb.get("callback_count") is not None and cb.get("exited_count") is not None:
                if int(cb["exited_count"]) > int(cb["callback_count"]):
                    problems.append({
                        "kind": "callback_count",
                        "task_id": cb.get("task_id"),
                        "callback_count": cb.get("callback_count"),
                        "exited_count": cb.get("exited_count"),
                    })

        if problems:
            status = (OracleStatus.VIOLATION
                      if oracle_contract == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = ("callback/generator lifecycle property failed"
                      if status == OracleStatus.VIOLATION
                      else "callback lifecycle deviation observed, but contract is unknown")
            return OracleEntry(
                self.oracle_id, status, reason,
                evidence={"problems": problems[:20], "checked": checked,
                          "contract_status": oracle_contract.value},
            )
        return OracleEntry(
            self.oracle_id, OracleStatus.PASS,
            "callback enter/exit, exception propagation and post-exception pipeline hold",
            evidence={"checked": checked},
        )
