"""O4: input and shared read-only state preservation.

The template records SHA-256 digests for every reconstructed target-call
argument before and after the call.  O4 consumes those records and reports a
violation only when a digest changes.  Mutable shared resources are *not*
judged here; they need an operation-level contract and are handled by the
resource-lifecycle backend.
"""
from __future__ import annotations

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


class InputPreservationOracle(BaseOracle):
    oracle_id = "O4"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O4 not in plan")
        oracle_contract = self.contract_status(plan)
        if oracle_contract == ContractStatus.UNSUPPORTED:
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE,
                               "contract marks this scenario unsupported")
        if observation.execution_status != ExecutionStatus.COMPLETED:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"execution status is {observation.execution_status.value}",
                evidence={"execution_status": observation.execution_status.value},
            )
        if not observation.observation_complete:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               "observation is incomplete")

        checked = 0
        diffs: List[Dict[str, Any]] = []
        missing = 0
        for task in observation.tasks or []:
            meta = task.metadata or {}
            for key in ("input_preservation", "shared_state_preservation"):
                record = meta.get(key)
                if not isinstance(record, dict):
                    continue
                checked += 1
                if record.get("preserved") is False:
                    diffs.append({
                        "task_id": task.task_id,
                        "input_id": task.input_id,
                        "source": key,
                        "diffs": record.get("diffs", [])[:20],
                    })
            if not isinstance(meta.get("input_preservation"), dict) and \
               not isinstance(meta.get("shared_state_preservation"), dict):
                missing += 1

        if checked == 0:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "no input-preservation observation was recorded "
                "(template did not emit input hashes)",
                evidence={"tasks": len(observation.tasks or []), "missing_tasks": missing},
            )
        if diffs:
            status = (OracleStatus.VIOLATION
                      if oracle_contract == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = ("input/shared read-only state was mutated during the call"
                      if status == OracleStatus.VIOLATION
                      else "input mutation observed, but contract is unknown")
            return OracleEntry(
                self.oracle_id, status, reason,
                evidence={"diffs": diffs[:20], "checked": checked,
                          "contract_status": oracle_contract.value},
            )
        return OracleEntry(
            self.oracle_id, OracleStatus.PASS,
            "all checked inputs/shared read-only state remained unchanged",
            evidence={"checked": checked},
        )
