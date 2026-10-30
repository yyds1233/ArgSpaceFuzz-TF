"""Oracle base interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..schemas import ConcurrentTestPlan, ExecutionObservation, OracleEntry


class BaseOracle(ABC):
    oracle_id: str = ""

    def is_applicable(self, plan: ConcurrentTestPlan, observation: ExecutionObservation) -> bool:
        return self.oracle_id in (plan.oracles or [])

    @abstractmethod
    def evaluate(
        self,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        context: Optional[Dict[str, Any]] = None,
    ) -> OracleEntry:
        ...
