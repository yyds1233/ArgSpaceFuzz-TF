"""LLM-assisted suitability analysis (P1 placeholder).

P0 deliberately uses rules only.  The class keeps the interface required by the
design document so that P1 can fill in the prompt and response handling without
changing the orchestrator.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .schemas import HarnessAnalysis, HarnessArtifact


class LLMSuitabilityAnalyzer:
    PROMPT_VERSION = "suitability-v1"

    def __init__(self, client: Any = None, model: Optional[str] = None,
                 model_config: Optional[Dict[str, Any]] = None):
        self.client = client
        self.model = model
        self.model_config = model_config or {}

    def analyze(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
        rule_result: Dict[str, Any],
        contract_info: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if self.client is None:
            return None
        raise NotImplementedError("LLM suitability analysis is scheduled for P1")

    def build_prompt(self, context: Dict[str, Any]) -> str:
        raise NotImplementedError("LLM suitability analysis is scheduled for P1")

    def parse_response(self, response: str) -> Dict[str, Any]:
        raise NotImplementedError("LLM suitability analysis is scheduled for P1")

    def validate_response(self, result: Dict[str, Any]) -> None:
        raise NotImplementedError("LLM suitability analysis is scheduled for P1")
