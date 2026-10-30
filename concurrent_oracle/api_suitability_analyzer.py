"""Suitability analysis orchestration: rules + optional LLM + contracts."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import DECISIONS, ORACLE_IDS, SCENARIO_IDS, STATE_MODELS
from .contract_registry import ContractRegistry
from .schemas import (
    CandidateScenario,
    ContractStatus,
    Decision,
    HarnessAnalysis,
    HarnessArtifact,
    SuitabilityReport,
    validate_oracles,
)


class APISuitabilityAnalyzer:
    def __init__(
        self,
        rules,
        llm_client: Any = None,
        contract_registry: Optional[ContractRegistry] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_analyzer: Any = None,
    ):
        self.rules = rules
        self.llm_client = llm_client
        self.contract_registry = contract_registry or ContractRegistry()
        self.config = config or {}
        self.llm_analyzer = llm_analyzer
        self.model_config = self.config.get("model_config", {})

    def analyze(self, artifact: HarnessArtifact, analysis: HarnessAnalysis) -> SuitabilityReport:
        rule_result = self.rules.analyze(artifact, analysis)
        llm_result: Optional[Dict[str, Any]] = None
        llm_error: Optional[str] = None

        if self.llm_analyzer is not None and self.llm_client is not None:
            try:
                contract_info = self.contract_registry.get_contract(artifact.api_name)
                llm_result = self.llm_analyzer.analyze(artifact, analysis, rule_result, contract_info)
                if llm_result is not None:
                    self.validate_llm_result(llm_result)
            except Exception as exc:
                llm_error = f"{type(exc).__name__}: {exc}"
                llm_result = None

        merged = self.merge_candidates(rule_result, llm_result)
        scenarios = merged["scenarios"]
        contract_status = self.contract_registry.aggregate_status(
            artifact.api_name, [s["scenario_id"] for s in scenarios]
        )
        decision, required_info = self._decide(
            artifact, analysis, scenarios, contract_status, merged, llm_error
        )

        categories = sorted(set(rule_result.get("api_categories", [])) |
                            set((llm_result or {}).get("api_categories", [])))
        rationale_parts = list(rule_result.get("reasons") or [])
        if llm_result and llm_result.get("rationale"):
            rationale_parts.append(str(llm_result["rationale"]))
        if llm_error:
            rationale_parts.append(f"LLM suitability failed; rules retained: {llm_error}")

        return SuitabilityReport(
            artifact_id=artifact.artifact_id,
            decision=decision,
            api_categories=categories,
            rule_candidates=list(rule_result.get("rule_candidates") or []),
            llm_candidates=list((llm_result or {}).get("oracles") or []),
            candidate_scenarios=[
                CandidateScenario(
                    scenario_id=s["scenario_id"],
                    oracles=list(s["oracles"]),
                    state_model=s["state_model"],
                    priority=s["priority"],
                    reason=s.get("reason", ""),
                )
                for s in scenarios
            ],
            contract_status=ContractStatus(contract_status),
            required_information=required_info,
            transformation_difficulty=merged.get("transformation_difficulty", "MEDIUM"),
            rationale="; ".join(rationale_parts),
            confidence=(llm_result or {}).get("confidence", 0.7 if scenarios else 0.0),
            rule_result=rule_result,
            llm_result=llm_result,
            model_config=self.model_config,
            prompt_version=getattr(self.llm_analyzer, "PROMPT_VERSION", None),
            raw_llm_response=(llm_result or {}).get("_raw_response"),
        )

    def merge_candidates(self, rule_result: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        scenarios: Dict[str, Dict[str, Any]] = {}
        for s in rule_result.get("candidate_scenarios") or []:
            sid = s.get("scenario_id")
            if sid not in SCENARIO_IDS:
                continue
            scenarios[sid] = {
                "scenario_id": sid,
                "oracles": list(s.get("oracles") or []),
                "state_model": s.get("state_model", "INDEPENDENT_INPUTS"),
                "priority": s.get("priority", "MEDIUM"),
                "reason": s.get("reason", ""),
            }

        if llm_result:
            for s in llm_result.get("candidate_scenarios") or []:
                sid = s.get("scenario_id")
                if sid not in SCENARIO_IDS:
                    raise ValueError(f"LLM returned unknown scenario id: {sid}")
                oracles = validate_oracles(s.get("oracles") or [])
                state_model = s.get("state_model", "INDEPENDENT_INPUTS")
                if state_model not in STATE_MODELS:
                    raise ValueError(f"LLM returned unknown state model: {state_model}")
                if sid in scenarios:
                    scenarios[sid]["oracles"] = sorted(set(scenarios[sid]["oracles"]) | set(oracles))
                    scenarios[sid]["reason"] = scenarios[sid].get("reason") or s.get("reason", "")
                else:
                    scenarios[sid] = {
                        "scenario_id": sid,
                        "oracles": oracles,
                        "state_model": state_model,
                        "priority": s.get("priority", "MEDIUM"),
                        "reason": s.get("reason", ""),
                    }

        for s in scenarios.values():
            validate_oracles(s["oracles"])
            if s["state_model"] not in STATE_MODELS:
                raise ValueError(f"unknown state model: {s['state_model']}")

        difficulty = "MEDIUM"
        if llm_result:
            difficulty = str(llm_result.get("transformation_difficulty", difficulty))
        elif all(s["state_model"] == "INDEPENDENT_INPUTS" for s in scenarios.values()):
            difficulty = "LOW"

        return {
            "scenarios": list(scenarios.values()),
            "transformation_difficulty": difficulty,
        }

    def validate_llm_result(self, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            raise ValueError("LLM result must be a JSON object")
        decision = result.get("decision")
        if decision is not None and decision not in DECISIONS:
            raise ValueError(f"LLM returned invalid decision: {decision}")
        scenarios = result.get("candidate_scenarios", [])
        if not isinstance(scenarios, list):
            raise ValueError("candidate_scenarios must be a list")
        for s in scenarios:
            if not isinstance(s, dict):
                raise ValueError("candidate scenario must be an object")
            if s.get("scenario_id") not in SCENARIO_IDS:
                raise ValueError(f"unknown scenario id: {s.get('scenario_id')}")
            validate_oracles(s.get("oracles") or [])
            if s.get("state_model") not in STATE_MODELS:
                raise ValueError(f"unknown state model: {s.get('state_model')}")
        conf = result.get("confidence")
        if conf is not None and not (0.0 <= float(conf) <= 1.0):
            raise ValueError("confidence must be in [0, 1]")

    def _decide(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
        scenarios: List[Dict[str, Any]],
        contract_status: str,
        merged: Dict[str, Any],
        llm_error: Optional[str],
    ) -> tuple[Decision, List[str]]:
        required: List[str] = []
        if not analysis.has_target_call:
            return Decision.INSUFFICIENT_INFO, ["target API call not found in AST"]
        if not scenarios:
            return Decision.INSUFFICIENT_INFO, ["no candidate concurrency scenario"]
        if contract_status == "UNSUPPORTED":
            return Decision.LOW_PRIORITY, ["contract marks this API/scenario unsupported"]
        if contract_status == "UNKNOWN":
            required.append("verify API contract for the selected scenario")
            return Decision.CONDITIONAL, required
        priorities = {s.get("priority", "MEDIUM") for s in scenarios}
        if priorities == {"LOW"}:
            return Decision.LOW_PRIORITY, required
        return Decision.READY, required
