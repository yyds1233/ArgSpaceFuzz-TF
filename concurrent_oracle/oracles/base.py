"""Oracle base interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    OracleEntry,
)


class BaseOracle(ABC):
    oracle_id: str = ""

    def is_applicable(self, plan: ConcurrentTestPlan, observation: ExecutionObservation) -> bool:
        return self.oracle_id in (plan.oracles or [])

    def contract_status(self, plan: ConcurrentTestPlan) -> ContractStatus:
        raw = (plan.oracle_contracts or {}).get(self.oracle_id)
        if raw:
            try:
                return ContractStatus(str(raw))
            except ValueError:
                pass
        return plan.contract_status

    @abstractmethod
    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        ...
