"""Oracle registry and unified result aggregation."""
from __future__ import annotations

import traceback
from typing import Any, Dict, Iterable, List, Optional

from .oracles.base import BaseOracle
from .oracles.consistency import ConsistencyOracle
from .oracles.independence import IndependenceOracle
from .schemas import (
    ConcurrentTestPlan,
    ExecutionObservation,
    ExecutionStatus,
    OracleEntry,
    OracleResult,
    OracleStatus,
)


class OracleEngine:
    def __init__(self, oracles: Optional[Iterable[BaseOracle]] = None):
        self.oracles: Dict[str, BaseOracle] = {}
        if oracles is None:
            oracles = [ConsistencyOracle(), IndependenceOracle()]
        for oracle in oracles:
            self.register(oracle)

    def register(self, oracle: BaseOracle) -> None:
        if not getattr(oracle, "oracle_id", None):
            raise ValueError("oracle must define oracle_id")
        self.oracles[oracle.oracle_id] = oracle

    def evaluate(
        self,
        observation: ExecutionObservation,
        plan: ConcurrentTestPlan,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleResult:
        entries: List[OracleEntry] = []
        failure_artifacts: List[str] = []
        for oid, oracle in sorted(self.oracles.items()):
            try:
                if not oracle.is_applicable(plan, observation):
                    entries.append(OracleEntry(oid, OracleStatus.NOT_APPLICABLE,
                                               "oracle not selected by the plan"))
                    continue
                entry = oracle.evaluate(plan, observation, context)
                entries.append(entry)
                if entry.status == OracleStatus.VIOLATION:
                    for key in ("stdout", "stderr"):
                        p = (observation.diagnostics or {}).get(key)
                        if p:
                            failure_artifacts.append(str(p))
            except Exception as exc:
                entries.append(OracleEntry(
                    oid, OracleStatus.INCONCLUSIVE,
                    f"oracle raised {type(exc).__name__}: {exc}",
                    evidence={"traceback": traceback.format_exc()},
                ))
        if observation.execution_status != ExecutionStatus.COMPLETED:
            # Keep explicit not-applicable entries, but force applicable entries
            # to INCONCLUSIVE when the execution itself did not complete.
            for e in entries:
                if e.status not in (OracleStatus.NOT_APPLICABLE,):
                    if e.status == OracleStatus.PASS:
                        e.status = OracleStatus.INCONCLUSIVE
                        e.reason = f"execution status {observation.execution_status.value}: {e.reason}"
        # P0 implements O2/O3.  Keep every plan Oracle visible in the result so
        # that a missing implementation is explicit rather than silently dropped.
        registered = set(self.oracles.keys())
        for oid in sorted(set(plan.oracles or []) - registered):
            entries.append(OracleEntry(
                oid, OracleStatus.INCONCLUSIVE,
                f"oracle {oid} is not implemented in P0",
            ))
        return OracleResult(
            plan_id=plan.plan_id,
            scenario=plan.scenario,
            execution_status=observation.execution_status,
            oracle_results=entries,
            evidence_level="E1",
            reproduction_rate=None,
            failure_artifacts=sorted(set(failure_artifacts)),
            metadata={
                "config_id": observation.config_id,
                "run_id": observation.run_id,
                "mode": observation.mode,
                "trials": observation.trials,
                "baseline_run_id": observation.baseline_run_id,
                "test_run_id": observation.test_run_id,
                "contract_status": plan.contract_status.value,
                "task_progress": observation.task_progress,
                "observation_complete": observation.observation_complete,
                "baseline_runtime": observation.baseline_runtime,
            },
        )
