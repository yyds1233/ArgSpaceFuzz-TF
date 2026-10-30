"""Generate ConcurrentTestPlan objects from a SuitabilityReport."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .config import ModuleConfig
from .constants import RUNTIME_MATRIX, SCENARIO_ORACLES, SCENARIO_CAPABILITIES
from .contract_registry import ContractRegistry
from .schemas import (
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
        max_configs_per_scenario: int = 4,
        max_plans_per_artifact: int = 8,
    ):
        self.config = config or ModuleConfig()
        self.contract_registry = contract_registry or ContractRegistry()
        self.max_configs_per_scenario = max_configs_per_scenario
        self.max_plans_per_artifact = max_plans_per_artifact
        self.last_skipped: Dict[str, str] = {}

    def generate(self, suitability: SuitabilityReport, api_name: str = "") -> List[ConcurrentTestPlan]:
        enabled = set(self.config.oracle_enabled)
        trials = int(self.config.execution.trials)
        timeout = int(self.config.execution.timeout_seconds)
        self.last_skipped = {}

        # Build scenario candidates first, then allocate one configuration per
        # scenario before spending the remaining plan budget on extra configs.
        candidates: List[Tuple[Any, List[Dict[str, Any]], List[str], Dict[str, str], ContractStatus]] = []
        for scenario in suitability.candidate_scenarios:
            scenario_id = validate_scenario(scenario.scenario_id)
            if scenario_id not in _SCENARIO_CONFIGS:
                continue
            cap = SCENARIO_CAPABILITIES.get(scenario_id) or {}
            if not cap.get("implemented", False):
                self.last_skipped[scenario_id] = cap.get("reason", "no execution backend")
                continue
            oracles = self.select_oracles(scenario_id, scenario.oracles, enabled, api_name)
            if not oracles:
                continue
            cap_oracles = set(cap.get("oracles") or [])
            implemented_oracles = [o for o in oracles if not cap_oracles or o in cap_oracles]
            unimplemented_oracles = [o for o in oracles if cap_oracles and o not in cap_oracles]
            if not implemented_oracles:
                self.last_skipped[scenario_id] = "no implemented oracle for this scenario"
                continue
            oracles = implemented_oracles
            oracle_contracts = {
                oid: self.contract_registry.get_oracle_status(api_name, scenario_id, oid)
                for oid in oracles
            } if api_name else {oid: "UNKNOWN" for oid in oracles}
            plan_status = (self.contract_registry.plan_status(api_name, scenario_id, oracles)
                           if api_name else "UNKNOWN")
            if plan_status == "UNSUPPORTED":
                continue
            runtime_configs = self.generate_runtime_configs(scenario_id)
            if not runtime_configs:
                continue
            candidates.append((scenario, runtime_configs, oracles, oracle_contracts,
                               ContractStatus(plan_status)))

        ordered: List[Tuple[Any, Dict[str, Any], List[str], Dict[str, str], ContractStatus]] = []
        # First pass: one plan per applicable scenario, so an early scenario
        # cannot exhaust the budget and hide internal-parallel coverage.
        for scenario, configs, oracles, oracle_contracts, plan_status in candidates:
            ordered.append((scenario, configs[0], oracles, oracle_contracts, plan_status))
        # Second pass: additional configurations.
        for scenario, configs, oracles, oracle_contracts, plan_status in candidates:
            for runtime in configs[1: self.max_configs_per_scenario]:
                ordered.append((scenario, runtime, oracles, oracle_contracts, plan_status))

        plans: List[ConcurrentTestPlan] = []
        for scenario, runtime, oracles, oracle_contracts, plan_status in ordered:
            if len(plans) >= self.max_plans_per_artifact:
                break
            idx = len(plans)
            scenario_id = scenario.scenario_id
            plan_id = make_plan_id(suitability.artifact_id, scenario_id, idx)
            plans.append(ConcurrentTestPlan(
                plan_id=plan_id,
                artifact_id=suitability.artifact_id,
                scenario=scenario_id,
                runtime=runtime,
                state_model=scenario.state_model,
                oracles=oracles,
                contract_status=plan_status,
                oracle_contracts=oracle_contracts,
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
                    "unimplemented_oracles": unimplemented_oracles,
                },
            ))
        return plans

    def generate_runtime_configs(self, scenario: str, api_categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        ids = _SCENARIO_CONFIGS.get(scenario, ["C00"])
        allowed = {
            "external_threads": set(int(x) for x in self.config.runtime.external_threads),
            "intra_op_threads": set(int(x) for x in self.config.runtime.intra_op_threads),
            "inter_op_threads": set(int(x) for x in self.config.runtime.inter_op_threads),
            "tf_data_parallel_calls": set(int(x) for x in self.config.runtime.tf_data_parallel_calls),
        }
        det_values = [bool(x) for x in self.config.runtime.deterministic] or [True]
        out: List[Dict[str, Any]] = []
        # Determinism is an explicit test dimension: produce a distinct
        # configuration identity for each requested value.
        for det in det_values:
            for cid in ids:
                base = dict(RUNTIME_MATRIX.get(cid, RUNTIME_MATRIX["C00"]))
                # Filter fixed matrix entries against the explicit YAML lists.
                if any(base[k] not in allowed[k] for k in allowed):
                    continue
                base.update({
                    "device": "CPU",
                    "deterministic": det,
                    "config_id": cid if len(det_values) == 1 else f"{cid}_det{int(det)}",
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
