"""O6: tf.data integrity and progress.

The recipe backend emits one or more ``dataset`` records per task.  O6 checks
cardinality, element-ID multiset, value/ID association, deterministic order
(only when the recipe declares it) and the post-early-stop pipeline progress.
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


class TFDataIntegrityOracle(BaseOracle):
    oracle_id = "O6"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O6 not in plan")
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

        datasets: List[Dict[str, Any]] = []
        for task in observation.tasks or []:
            record = (task.metadata or {}).get("dataset")
            if isinstance(record, dict):
                datasets.append({"task_id": task.task_id, **record})
            records = (task.metadata or {}).get("datasets")
            if isinstance(records, list):
                for item in records:
                    if isinstance(item, dict):
                        datasets.append({"task_id": task.task_id, **item})
        if not datasets:
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "no tf.data observation was recorded",
                evidence={"tasks": len(observation.tasks or [])},
            )

        problems: List[Dict[str, Any]] = []
        checked = 0
        for ds in datasets:
            checked += 1
            expected_count = ds.get("expected_cardinality")
            observed_count = ds.get("observed_cardinality")
            if expected_count is not None and observed_count is not None \
                    and int(expected_count) != int(observed_count):
                problems.append({"kind": "cardinality", "task_id": ds.get("task_id"),
                                 "expected": expected_count, "observed": observed_count})
            expected_ids = ds.get("expected_ids")
            observed_ids = ds.get("observed_ids")
            if isinstance(expected_ids, list) and isinstance(observed_ids, list):
                missing = list((Counter(map(str, expected_ids)) -
                                Counter(map(str, observed_ids))).elements())
                extra = list((Counter(map(str, observed_ids)) -
                              Counter(map(str, expected_ids))).elements())
                if missing or extra:
                    problems.append({"kind": "id_multiset", "task_id": ds.get("task_id"),
                                     "missing": missing[:20], "extra": extra[:20]})
            mismatched = ds.get("mismatched_ids") or ds.get("value_id_mismatch")
            if mismatched:
                problems.append({"kind": "value_id", "task_id": ds.get("task_id"),
                                 "mismatched": list(mismatched)[:20]})
            if ds.get("deterministic") and ds.get("order_preserved") is False:
                problems.append({"kind": "order", "task_id": ds.get("task_id"),
                                 "first_expected": (ds.get("expected_ids") or [])[:10],
                                 "first_observed": (ds.get("observed_ids") or [])[:10]})
            early = ds.get("early_stop")
            if isinstance(early, dict) and early.get("subsequent_pipeline_ok") is False:
                problems.append({"kind": "early_stop_recovery", "task_id": ds.get("task_id"),
                                 "detail": early})

        if problems:
            status = (OracleStatus.VIOLATION
                      if oracle_contract == ContractStatus.SUPPORTED
                      else OracleStatus.INCONCLUSIVE)
            reason = ("tf.data integrity/progress property failed"
                      if status == OracleStatus.VIOLATION
                      else "tf.data integrity deviation observed, but contract is unknown")
            return OracleEntry(
                self.oracle_id, status, reason,
                evidence={"problems": problems[:20], "checked": checked,
                          "contract_status": oracle_contract.value},
            )
        return OracleEntry(
            self.oracle_id, OracleStatus.PASS,
            "tf.data cardinality, IDs and declared order/progress properties hold",
            evidence={"checked": checked},
        )
