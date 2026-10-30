"""O3: call independence and task/result association."""
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
        if observation.execution_status != ExecutionStatus.COMPLETED:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                f"execution status is {observation.execution_status.value}",
                evidence={"execution_status": observation.execution_status.value},
            )
        if not observation.observation_complete:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE, "observation is incomplete")

        cfg = (context or {}).get("comparison") or {}
        expected = int(plan.runtime.get("external_threads", len(observation.tasks)))
        tasks = list(observation.tasks)
        problems: List[Dict[str, Any]] = []

        if len(tasks) != expected:
            problems.append({"kind": "task_count", "expected": expected, "actual": len(tasks)})
        ids = [t.task_id for t in tasks]
        if len(set(ids)) != len(ids):
            problems.append({"kind": "duplicate_task_ids", "task_ids": ids})
        if set(ids) != set(range(expected)):
            problems.append({"kind": "task_id_set", "expected": list(range(expected)), "actual": sorted(ids)})

        cross_talk = []
        checked = 0
        for task in tasks:
            baseline = (observation.baseline or {}).get(task.input_id)
            matches = find_cross_talk(task, observation.baseline or {}, cfg)
            if matches:
                cross_talk.append({"task_id": task.task_id, "input_id": task.input_id,
                                   "matches_other_inputs": matches})
                continue
            r = compare_task_to_baseline(task, baseline, cfg)
            if r.get("equal"):
                checked += 1
            elif not r.get("inconclusive"):
                problems.append({"kind": "output_association", "task_id": task.task_id,
                                 "input_id": task.input_id, "diffs": r.get("diffs", [])[:10]})

        if cross_talk:
            return OracleEntry(self.oracle_id, OracleStatus.VIOLATION,
                               "task output matches another task's input baseline (cross-task association)",
                               evidence={"cross_talk": cross_talk, "checked": checked})
        if problems:
            status = (OracleStatus.VIOLATION
                      if plan.contract_status == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            return OracleEntry(self.oracle_id, status,
                               f"{len(problems)} independence/association problem(s)",
                               evidence={"problems": problems[:20], "checked": checked,
                                         "contract_status": plan.contract_status.value})
        if checked != expected:
            return OracleEntry(self.oracle_id, OracleStatus.INCONCLUSIVE,
                               f"only {checked}/{expected} tasks produced comparable outputs",
                               evidence={"checked": checked, "expected": expected})
        return OracleEntry(self.oracle_id, OracleStatus.PASS,
                           "all tasks completed with outputs associated to their own inputs",
                           evidence={"checked": checked})
