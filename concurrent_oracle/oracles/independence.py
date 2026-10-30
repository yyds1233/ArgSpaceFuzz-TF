"""O3: call independence and task/result association.

Important: a match against another task's baseline is only *suspected*
cross-talk.  The task's own baseline is checked first; non-injective APIs can
legitimately produce identical outputs for different inputs.  Suspected
cross-talk is contract-gated and never automatically VIOLATION for UNKNOWN
contracts.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..result_comparator import compare_task_to_baseline, find_cross_talk
from ..schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    ExecutionStatus,
    OracleEntry,
    OracleStatus,
)
from .base import BaseOracle


class IndependenceOracle(BaseOracle):
    oracle_id = "O3"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O3 not in plan")
        if plan.contract_status == ContractStatus.UNSUPPORTED:
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

        cfg = (context or {}).get("comparison") or {}
        expected = int(plan.runtime.get("external_threads", len(observation.tasks)))
        tasks = list(observation.tasks)
        structural: List[Dict[str, Any]] = []

        if len(tasks) != expected:
            structural.append({"kind": "task_count", "expected": expected, "actual": len(tasks)})
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            structural.append({"kind": "duplicate_task_ids", "task_ids": ids})
        if set(ids) != set(range(expected)):
            structural.append({"kind": "task_id_set", "expected": list(range(expected)),
                               "actual": sorted(ids)})

        selected_ids = [t.input_id for t in tasks]
        checked = 0
        inconclusive_tasks = 0
        suspected: List[Dict[str, Any]] = []
        mismatches: List[Dict[str, Any]] = []

        for task in tasks:
            own_baseline = (observation.baseline or {}).get(task.input_id)
            own = compare_task_to_baseline(task, own_baseline, cfg)
            if own.get("inconclusive"):
                inconclusive_tasks += 1
                continue
            if own.get("equal"):
                checked += 1
                continue
            matches = find_cross_talk(
                task, observation.baseline or {}, cfg,
                candidate_input_ids=selected_ids,
                own_baseline=own_baseline,
            )
            if matches:
                suspected.append({
                    "kind": "suspected_cross_talk",
                    "task_id": task.task_id,
                    "input_id": task.input_id,
                    "matches_other_inputs": matches,
                    "own_diffs": own.get("diffs", [])[:10],
                })
            else:
                mismatches.append({
                    "kind": "output_association",
                    "task_id": task.task_id,
                    "input_id": task.input_id,
                    "diffs": own.get("diffs", [])[:10],
                })

        if structural:
            return OracleEntry(
                self.oracle_id, OracleStatus.VIOLATION,
                f"{len(structural)} structural independence problem(s)",
                evidence={"problems": structural[:20], "checked": checked},
            )

        if suspected:
            status = (OracleStatus.VIOLATION
                      if plan.contract_status == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = ("task output matches another selected task's baseline "
                      "(suspected cross-task association)")
            if status == OracleStatus.INCONCLUSIVE:
                reason += "; contract is unknown, evidence retained for review"
            return OracleEntry(self.oracle_id, status, reason,
                               evidence={"suspected_cross_talk": suspected[:20],
                                         "checked": checked,
                                         "contract_status": plan.contract_status.value})

        if mismatches:
            status = (OracleStatus.VIOLATION
                      if plan.contract_status == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            return OracleEntry(self.oracle_id, status,
                               f"{len(mismatches)} output/association mismatch(es)",
                               evidence={"problems": mismatches[:20], "checked": checked,
                                         "contract_status": plan.contract_status.value})

        if checked != expected:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"only {checked}/{expected} tasks produced comparable outputs",
                evidence={"checked": checked, "inconclusive_tasks": inconclusive_tasks},
            )
        return OracleEntry(self.oracle_id, OracleStatus.PASS,
                           "all tasks completed with outputs associated to their own inputs",
                           evidence={"checked": checked})
