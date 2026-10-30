"""Framework adapter abstraction."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class FrameworkAdapter(ABC):
    @abstractmethod
    def validate_environment(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def configure_runtime(self, config: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def load_case_inputs(self, case_path: str):
        ...

    @abstractmethod
    def materialize_output(self, output):
        ...

    @abstractmethod
    def compare_outputs(self, expected, actual, config: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def cleanup(self) -> None:
        ...
