"""O5: exception isolation and recovery.

The exception-recovery backend runs three phases in one process:

  A: a normal call
  B: a deliberately invalid call (expected input exception)
  C: another normal call after B

O5 checks that B is rejected with an expected input exception, A completes,
and C still completes.  A/C failures after B are evidence of an isolation or
recovery problem; they remain INCONCLUSIVE when the API contract is UNKNOWN.
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


class ExceptionRecoveryOracle(BaseOracle):
    oracle_id = "O5"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O5 not in plan")
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

        expected = self._expected_exceptions(plan)
        by_phase: Dict[str, List[Any]] = {"A": [], "B": [], "C": []}
        for task in observation.tasks or []:
            phase = str((task.metadata or {}).get("phase", "")).upper()
            if phase in by_phase:
                by_phase[phase].append(task)

        missing = [p for p, tasks in by_phase.items() if not tasks]
        if missing:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"missing phase(s): {', '.join(missing)}",
                evidence={"phase_counts": {p: len(v) for p, v in by_phase.items()}},
            )

        phase_a = by_phase["A"]
        phase_b = by_phase["B"]
        phase_c = by_phase["C"]

        b_problems: List[Dict[str, Any]] = []
        for task in phase_b:
            record = (task.metadata or {}).get("invalid_input") or {}
            observed = record.get("observed_exception") or task.exception_type
            if observed not in expected:
                b_problems.append({
                    "task_id": task.task_id,
                    "observed_exception": observed,
                    "expected": sorted(expected),
                })
        a_failed = [t for t in phase_a if t.status != "COMPLETED"]
        c_failed = [t for t in phase_c if t.status != "COMPLETED"]

        if b_problems:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "phase B did not produce an expected input rejection; "
                "the recovery test is not meaningful",
                evidence={"phase_b_problems": b_problems[:20]},
            )
        if a_failed or c_failed:
            status = (OracleStatus.VIOLATION
                      if oracle_contract == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = ("normal task failed after/beside an expected exception"
                      if status == OracleStatus.VIOLATION
                      else "recovery failure observed, but contract is unknown")
            return OracleEntry(
                self.oracle_id, status, reason,
                evidence={
                    "phase_a_failed": [{"task_id": t.task_id, "status": t.status,
                                        "exception": t.exception_type} for t in a_failed][:20],
                    "phase_c_failed": [{"task_id": t.task_id, "status": t.status,
                                        "exception": t.exception_type} for t in c_failed][:20],
                    "contract_status": oracle_contract.value,
                },
            )
        return OracleEntry(
            self.oracle_id, OracleStatus.PASS,
            "expected exception was isolated and the following task completed",
            evidence={"phase_counts": {p: len(v) for p, v in by_phase.items()}},
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
