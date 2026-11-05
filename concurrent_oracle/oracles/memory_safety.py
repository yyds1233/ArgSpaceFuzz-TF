"""O9: CPU kernel memory safety via ASan/TSan/UBSan reports.

O9 consumes the sanitizer record produced by the runner.  A parsed sanitizer
report is concrete evidence (VIOLATION); a completed run under an enabled
instrumented build is PASS; a disabled/uninstrumented build is INCONCLUSIVE.
"""
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


class MemorySafetyOracle(BaseOracle):
    oracle_id = "O9"

    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        if not self.is_applicable(plan, observation):
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE, "O9 not in plan")
        oracle_contract = self.contract_status(plan)
        if oracle_contract == ContractStatus.UNSUPPORTED:
            return OracleEntry(self.oracle_id, OracleStatus.NOT_APPLICABLE,
                               "contract marks this scenario unsupported")
        sanitizer = (observation.diagnostics or {}).get("sanitizer") or {}
        if not isinstance(sanitizer, dict) or not sanitizer.get("enabled"):
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "sanitizer is not enabled for this run; O9 has no evidence",
                evidence={"sanitizer": sanitizer},
            )
        evidence = {
            "sanitizer": sanitizer,
            "execution_status": observation.execution_status.value,
            "process_exit_code": observation.process_exit_code,
            "process_signal": observation.process_signal,
            "stderr": observation.stderr_path,
        }
        if sanitizer.get("detected"):
            return OracleEntry(
                self.oracle_id, OracleStatus.VIOLATION,
                f"{sanitizer.get('report_type') or 'sanitizer'} report detected: "
                f"{sanitizer.get('summary') or 'see evidence'}",
                evidence=evidence,
            )
        if not sanitizer.get("instrumented", True):
            return OracleEntry(
                self.oracle_id, OracleStatus.INCONCLUSIVE,
                "sanitizer is enabled but the build/run is not verified as instrumented",
                evidence=evidence,
            )
        if observation.execution_status == ExecutionStatus.COMPLETED:
            return OracleEntry(
                self.oracle_id, OracleStatus.PASS,
                "no sanitizer report was detected in a completed instrumented run",
                evidence=evidence,
            )
        return OracleEntry(
            self.oracle_id, OracleStatus.INCONCLUSIVE,
            f"execution ended with {observation.execution_status.value} without a "
            "parsed sanitizer report",
            evidence=evidence,
        )
