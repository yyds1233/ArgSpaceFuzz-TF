"""O2: serial/parallel semantic consistency."""
from __future__ import annotations

from typing import Any, Dict, Optional

from ..result_comparator import compare_task_to_baseline
from ..schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    ExecutionStatus,
    OracleEntry,
    OracleStatus,
)
from .base import BaseOracle


class ConsistencyOracle(BaseOracle):
    oracle_id = "O2"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O2 not in plan")
        oracle_contract = self.contract_status(plan)
        if oracle_contract == ContractStatus.UNSUPPORTED:
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE,
                               "contract marks this scenario unsupported")
        if observation.execution_status != ExecutionStatus.COMPLETED:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"execution status is {observation.execution_status.value}; cannot compare",
                evidence={"execution_status": observation.execution_status.value},
            )
        if not observation.observation_complete:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               "observation is incomplete")
        if not observation.tasks:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE, "no task observations")

        cfg = (context or {}).get("comparison") or {}
        diffs = []
        checked = 0
        inconclusive = False
        for task in observation.tasks:
            baseline = (observation.baseline or {}).get(task.input_id)
            r = compare_task_to_baseline(task, baseline, cfg)
            if r.get("inconclusive"):
                inconclusive = True
                continue
            if r.get("equal"):
                checked += 1
            else:
                diffs.append({"task_id": task.task_id, "input_id": task.input_id,
                              "diffs": r.get("diffs", [])[:10], "reason": r.get("reason")})

        if diffs:
            status = (OracleStatus.VIOLATION
                      if oracle_contract == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = (f"{len(diffs)} task(s) differ from serial baseline"
                      if status == OracleStatus.VIOLATION
                      else "serial/parallel difference observed, but API contract is unknown")
            return OracleEntry(self.oracle_id, status, reason,
                               evidence={"diffs": diffs[:20], "checked": checked,
                                         "contract_status": oracle_contract.value})
        if checked == 0:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               "no task produced a comparable output")
        if inconclusive:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               "some tasks had incomplete observations",
                               evidence={"checked": checked})
        return OracleEntry(self.oracle_id, OracleStatus.PASS,
                           "all completed tasks match the serial baseline",
                           evidence={"checked": checked})
