"""Generate ConcurrentTestPlan objects from a SuitabilityReport."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import ModuleConfig
from .constants import RUNTIME_MATRIX, SCENARIO_ORACLES
from .contract_registry import ContractRegistry
from .schemas import (
    CandidateScenario,
    ConcurrentTestPlan,
    ContractStatus,
    SuitabilityReport,
    make_plan_id,
    validate_oracles,
    validate_scenario,
)


_SCENARIO_CONFIGS = {
    "independent_api_calls": ["C03", "C04"],
    "shared_readonly_input": ["C03", "C04"],
    "shared_resource_access": ["C03"],
    "parallel_kernel_execution": ["C01", "C02"],
    "tfdata_parallel_map": ["C06", "C07"],
    "tfdata_interleave": ["C06"],
    "tfdata_early_stop": ["C06"],
    "python_callback_lifecycle": ["C03", "C04"],
    "exception_recovery": ["C03"],
    "invalid_input_cpu": ["C00"],
}

_EXPECTED_RELATION = {
    "independent_api_calls": "SERIAL_PARALLEL_EQUIVALENCE_AND_TASK_ASSOCIATION",
    "shared_readonly_input": "READONLY_INPUT_PRESERVED_AND_TASK_ASSOCIATION",
    "shared_resource_access": "SHARED_RESOURCE_STATE_CONTRACT",
    "parallel_kernel_execution": "SERIAL_PARALLEL_EQUIVALENCE_AND_PROGRESS",
    "tfdata_parallel_map": "DATA_INTEGRITY_AND_ORDER",
    "tfdata_interleave": "DATA_INTEGRITY_AND_ORDER",
    "tfdata_early_stop": "PIPELINE_STOP_AND_RESOURCE_RELEASE",
    "python_callback_lifecycle": "CALLBACK_AND_RESOURCE_LIFECYCLE",
    "exception_recovery": "EXCEPTION_ISOLATION_AND_RECOVERY",
    "invalid_input_cpu": "SAFE_REJECTION_AND_MEMORY_SAFETY",
}


class ScenarioGenerator:
    def __init__(
        self,
        config: Optional[ModuleConfig] = None,
        contract_registry: Optional[ContractRegistry] = None,
        max_configs_per_scenario: int = 2,
        max_plans_per_artifact: int = 4,
    ):
        self.config = config or ModuleConfig()
        self.contract_registry = contract_registry or ContractRegistry()
        self.max_configs_per_scenario = max_configs_per_scenario
        self.max_plans_per_artifact = max_plans_per_artifact

    def generate(self, suitability: SuitabilityReport, api_name: str = "") -> List[ConcurrentTestPlan]:
        plans: List[ConcurrentTestPlan] = []
        enabled = set(self.config.oracle_enabled)
        trials = int(self.config.execution.trials)
        timeout = int(self.config.execution.timeout_seconds)

        for scenario in suitability.candidate_scenarios:
            scenario_id = validate_scenario(scenario.scenario_id)
            if scenario_id not in _SCENARIO_CONFIGS:
                continue
            oracles = self.select_oracles(scenario_id, scenario.oracles, enabled, api_name)
            if not oracles:
                continue
            runtime_configs = self.generate_runtime_configs(scenario_id)
            for runtime in runtime_configs[: self.max_configs_per_scenario]:
                idx = len(plans)
                plan_id = make_plan_id(suitability.artifact_id, scenario_id, idx)
                plans.append(ConcurrentTestPlan(
                    plan_id=plan_id,
                    artifact_id=suitability.artifact_id,
                    scenario=scenario_id,
                    runtime=runtime,
                    state_model=scenario.state_model,
                    oracles=oracles,
                    contract_status=suitability.contract_status,
                    trials=trials,
                    timeout_seconds=timeout,
                    case_source=f"captured_cases/{plan_id}",
                    expected_relation=_EXPECTED_RELATION.get(scenario_id, "GENERIC_OBSERVATION"),
                    metadata={
                        "api_name": api_name,
                        "priority": scenario.priority,
                        "reason": scenario.reason,
                        "config_id": runtime.get("config_id"),
                        "api_categories": list(suitability.api_categories),
                    },
                ))
                if len(plans) >= self.max_plans_per_artifact:
                    return plans
        return plans

    def generate_runtime_configs(self, scenario: str, api_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        ids = _SCENARIO_CONFIGS.get(scenario, ["C00"])
        out: List[Dict[str, Any]] = []
        for cid in ids:
            base = dict(RUNTIME_MATRIX.get(cid, RUNTIME_MATRIX["C00"]))
            base.update({
                "device": "CPU",
                "deterministic": True,
                "config_id": cid,
            })
            out.append(base)
        return out

    def select_oracles(
        self,
        scenario: str,
        candidate_oracles: Optional[List[str]] = None,
        enabled: Optional[set] = None,
        api_name: str = "",
    ) -> List[str]:
        base = list(candidate_oracles or SCENARIO_ORACLES.get(scenario, ["O3"]))
        if enabled is not None:
            base = [o for o in base if o in enabled]
        if api_name:
            filtered = []
            for oracle in base:
                info = self.contract_registry.validate_oracle_applicability(api_name, scenario, oracle)
                if info.get("applicable", True):
                    filtered.append(oracle)
            base = filtered
        validate_oracles(base)
        return sorted(set(base))
