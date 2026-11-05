"""O1: invalid-input safe rejection.

The invalid-input backend mutates one captured argument (None/wrong kind/
negative shape/etc.) and records the observed target exception.  O1 checks
that the API rejects the invalid input with a known input-validation error
instead of accepting it, crashing natively, or hanging.  Native memory-safety
findings are O9's responsibility.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from ..constants import DEFAULT_INVALID_INPUT_EXCEPTIONS
from ..schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    ExecutionStatus,
    OracleEntry,
    OracleStatus,
)
from .base import BaseOracle


class InvalidInputOracle(BaseOracle):
    oracle_id = "O1"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O1 not in plan")
        oracle_contract = self.contract_status(plan)
        if oracle_contract == ContractStatus.UNSUPPORTED:
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE,
                               "contract marks this scenario unsupported")
        if observation.execution_status in (ExecutionStatus.TIMEOUT,
                                            ExecutionStatus.CRASH,
                                            ExecutionStatus.SETUP_ERROR):
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"execution ended with {observation.execution_status.value}; "
                "native safety is reported by O9",
                evidence={"execution_status": observation.execution_status.value},
            )
        if observation.execution_status != ExecutionStatus.COMPLETED:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               f"execution status is {observation.execution_status.value}")

        expected = self._expected_exceptions(plan)
        checked = 0
        rejected: List[Dict[str, Any]] = []
        not_rejected: List[Dict[str, Any]] = []
        unexpected: List[Dict[str, Any]] = []
        for task in observation.tasks or []:
            meta = task.metadata or {}
            record = meta.get("invalid_input")
            if not isinstance(record, dict):
                continue
            checked += 1
            observed = record.get("observed_exception") or task.exception_type
            accepted = bool(record.get("accepted"))
            if observed and observed in expected:
                rejected.append({"task_id": task.task_id, "input_id": task.input_id,
                                 "exception": observed})
            elif observed:
                unexpected.append({"task_id": task.task_id, "input_id": task.input_id,
                                   "exception": observed,
                                   "expected": sorted(expected)})
            elif accepted or task.status == "COMPLETED":
                not_rejected.append({"task_id": task.task_id, "input_id": task.input_id,
                                     "status": task.status})
            else:
                unexpected.append({"task_id": task.task_id, "input_id": task.input_id,
                                   "exception": None, "status": task.status})

        if checked == 0:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "no invalid-input observation was recorded",
                evidence={"tasks": len(observation.tasks or [])},
            )
        problems = not_rejected + unexpected
        if problems:
            status = (OracleStatus.VIOLATION
                      if oracle_contract == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = ("invalid input was accepted or rejected with an unexpected "
                      "exception" if status == OracleStatus.VIOLATION
                      else "invalid-input behavior deviates, but contract is unknown")
            return OracleEntry(
                self.oracle_id, status, reason,
                evidence={"rejected": rejected[:20], "problems": problems[:20],
                          "expected_exceptions": sorted(expected),
                          "contract_status": oracle_contract.value},
            )
        return OracleEntry(
            self.oracle_id, OracleStatus.PASS,
            "all invalid inputs were rejected with an expected input exception",
            evidence={"rejected": rejected[:20], "checked": checked},
        )

    @staticmethod
    def _expected_exceptions(plan: ConcurrentTestPlan) -> set[str]:
        raw = (plan.metadata or {}).get("expected_exception_types")
        if isinstance(raw, str):
            return {raw}
        if isinstance(raw, Iterable):
            vals = {str(x) for x in raw}
            if vals:
                return vals
        return set(DEFAULT_INVALID_INPUT_EXCEPTIONS)
