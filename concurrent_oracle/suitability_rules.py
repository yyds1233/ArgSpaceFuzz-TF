"""Rule-based pre-screening for TensorFlow concurrency suitability."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .constants import ORACLE_IDS, SCENARIO_ORACLES
from .schemas import HarnessAnalysis, HarnessArtifact, validate_oracles, validate_state_model

# API leaf names that are known to be CPU-heavy / parallel-kernel candidates.
_CPU_PARALLEL_HINTS = {
    "conv2d", "conv3d", "conv1d", "convolution", "matmul", "batch_matmul",
    "einsum", "reduce_sum", "reduce_mean", "reduce_max", "reduce_min",
    "cumsum", "cumprod", "fft", "ifft", "fft2d", "fft3d", "stft", "top_k",
    "sort", "argsort", "unique", "non_max_suppression", "resize", "resize_bilinear",
    "resize_nearest_neighbor", "depthwise_conv2d", "separable_conv2d",
}
_RANDOM_HINTS = {"random", "stateless", "generator", "seed"}
_IO_HINTS = {"read_file", "write_file", "decode", "encode", "io", "dataset", "tfrecord"}


class SuitabilityRules:
    """Rule engine that only produces candidates; it never hard-rejects an API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_scenarios = int(self.config.get("max_scenarios", 4))

    def analyze(self, artifact: HarnessArtifact, analysis: HarnessAnalysis) -> Dict[str, Any]:
        categories: List[str] = []
        scenarios: List[Dict[str, Any]] = []
        reasons: List[str] = []

        if analysis.ast_parse_status != "SUCCESS":
            return {
                "api_categories": ["UNKNOWN"],
                "rule_candidates": [],
                "candidate_scenarios": [],
                "state_models": [],
                "reasons": ["AST parse failed"],
                "warnings": list(analysis.warnings),
            }

        if not analysis.has_target_call:
            categories.append("UNKNOWN")
            reasons.append(f"target API {artifact.api_name!r} not found in AST")

        has_dataset = self.detect_dataset_pattern(analysis)
        has_callback = self.detect_callback_pattern(analysis)
        has_generator = self.detect_generator_pattern(analysis)
        has_shared_resource = self.detect_shared_resource(analysis)
        has_lifecycle = self.detect_lifecycle(analysis)
        has_random, has_stateless = self.detect_random(analysis, artifact.api_name)
        has_thread_config = self.detect_thread_config(analysis)
        has_cpu_io = self.detect_cpu_io(analysis, artifact.api_name)
        is_pure = self.detect_pure_tensor(analysis, artifact.api_name)
        is_cpu_parallel = self.detect_cpu_parallel(artifact.api_name, analysis)

        if is_pure:
            categories.append("PURE_TENSOR_OP")
            self._add(scenarios, "independent_api_calls", "HIGH",
                      "Independent API calls with isolated inputs exercise concurrent execution.")
            self._add(scenarios, "shared_readonly_input", "MEDIUM",
                      "A read-only tensor can be shared across workers to check input preservation.")
        if is_cpu_parallel:
            categories.append("CPU_PARALLEL_OP")
            self._add(scenarios, "parallel_kernel_execution", "HIGH",
                      "The API likely enters a CPU-parallel kernel; intra-op/inter-op settings are relevant.")
        if has_dataset:
            categories.append("DATASET_PIPELINE")
            self._add(scenarios, "tfdata_parallel_map", "HIGH",
                      "Dataset pipeline detected; parallel map/interleave can be tested.")
            self._add(scenarios, "tfdata_early_stop", "MEDIUM",
                      "Dataset early-stop and iterator lifecycle can be tested.")
        if has_callback:
            categories.append("PYTHON_CALLBACK")
            self._add(scenarios, "python_callback_lifecycle", "HIGH",
                      "Python callback detected; GIL/callback lifecycle is relevant.")
        if has_generator:
            categories.append("GENERATOR")
            self._add(scenarios, "python_callback_lifecycle", "MEDIUM",
                      "Generator-backed dataset detected; producer/consumer lifecycle is relevant.")
        if has_shared_resource:
            categories.append("SHARED_VARIABLE")
            self._add(scenarios, "shared_resource_access", "MEDIUM",
                      "Shared TensorFlow resource detected; access isolation should be checked.")
        if has_random:
            categories.append("RANDOM_STATE")
            reasons.append("stateful random API detected; random differences are not automatically bugs")
        if has_stateless:
            categories.append("STATELESS_RANDOM")
            reasons.append("stateless random API detected; explicit seeds allow stronger expectations")
        if has_thread_config:
            categories.append("THREAD_CONFIG")
            self._add(scenarios, "parallel_kernel_execution", "MEDIUM",
                      "Explicit thread configuration detected.")
        if has_lifecycle:
            categories.append("RESOURCE_LIFECYCLE")
            self._add(scenarios, "exception_recovery", "LOW",
                      "Resource cleanup or session lifecycle detected.")
        if has_cpu_io:
            categories.append("CPU_IO")
            self._add(scenarios, "tfdata_early_stop", "LOW",
                      "CPU I/O path detected; parallel pipeline progress is relevant.")
        if not categories:
            categories.append("UNKNOWN")
            self._add(scenarios, "independent_api_calls", "LOW",
                      "No specific pattern matched; use a conservative independent-call test.")

        # Cap scenario count and attach oracles/state models.
        scenarios = self._dedupe(scenarios)[: self.max_scenarios]
        for s in scenarios:
            scenario_id = s["scenario_id"]
            s["oracles"] = list(SCENARIO_ORACLES.get(scenario_id, ["O3"]))
            s["state_model"] = self._state_model_for(scenario_id, has_shared_resource)
            validate_oracles(s["oracles"])
            validate_state_model(s["state_model"])

        rule_candidates = sorted({o for s in scenarios for o in s["oracles"]})
        return {
            "api_categories": sorted(set(categories)),
            "rule_candidates": rule_candidates,
            "candidate_scenarios": scenarios,
            "state_models": sorted({s["state_model"] for s in scenarios}),
            "reasons": reasons,
            "warnings": list(analysis.warnings),
            "has_dataset": has_dataset,
            "has_callback": has_callback,
            "has_generator": has_generator,
            "has_shared_resource": has_shared_resource,
            "has_lifecycle": has_lifecycle,
            "has_random": has_random,
            "has_stateless_random": has_stateless,
            "has_thread_config": has_thread_config,
            "has_cpu_io": has_cpu_io,
            "is_pure_tensor": is_pure,
            "is_cpu_parallel": is_cpu_parallel,
        }

    # ── detectors ──────────────────────────────────────────────

    def classify_api(self, api_name: str) -> List[str]:
        leaf = (api_name or "").split(".")[-1].lower()
        out = []
        if any(h in leaf for h in _CPU_PARALLEL_HINTS):
            out.append("CPU_PARALLEL_OP")
        if any(h in api_name.lower() for h in _RANDOM_HINTS):
            out.append("RANDOM_STATE")
        if "stateless" in api_name.lower():
            out.append("STATELESS_RANDOM")
        if any(h in api_name.lower() for h in _IO_HINTS):
            out.append("CPU_IO")
        return out

    def detect_dataset_pattern(self, analysis: HarnessAnalysis) -> bool:
        if analysis.dataset_operations:
            return True
        for imp in analysis.imports:
            if imp.get("module", "").startswith("tensorflow.data"):
                return True
        return False

    def detect_callback_pattern(self, analysis: HarnessAnalysis) -> bool:
        if analysis.callback_functions:
            return True
        for op in analysis.dataset_operations + analysis.tensor_construction:
            call = op.get("call", "")
            if call in {"tf.py_function", "tf.numpy_function"}:
                return True
        return False

    def detect_generator_pattern(self, analysis: HarnessAnalysis) -> bool:
        for op in analysis.dataset_operations + analysis.resource_lifecycle:
            if "from_generator" in op.get("call", ""):
                return True
        return any("generator" in name.lower() for name in analysis.callback_functions)

    def detect_shared_resource(self, analysis: HarnessAnalysis) -> bool:
        return bool(analysis.resource_variables)

    def detect_lifecycle(self, analysis: HarnessAnalysis) -> bool:
        return bool(analysis.cleanup_expressions) or bool(analysis.resource_lifecycle)

    def detect_random(self, analysis: HarnessAnalysis, api_name: str) -> tuple[bool, bool]:
        target_text = " ".join(c.api_name for c in analysis.target_calls)
        text = (api_name or "") + " " + " ".join(
            op.get("call", "") for op in analysis.tensor_construction
        ) + " " + target_text
        text = text.lower()
        has_random = "random" in text or "generator" in text
        has_stateless = "stateless" in text
        return has_random, has_stateless

    def detect_thread_config(self, analysis: HarnessAnalysis) -> bool:
        text = " ".join(op.get("call", "") for op in analysis.resource_lifecycle + analysis.tensor_construction)
        return "config.threading" in text or "set_intra_op_parallelism_threads" in text

    def detect_cpu_io(self, analysis: HarnessAnalysis, api_name: str) -> bool:
        target_text = " ".join(c.api_name for c in analysis.target_calls)
        text = (api_name or "") + " " + " ".join(
            op.get("call", "") for op in analysis.dataset_operations
        ) + " " + target_text
        return any(h in text.lower() for h in _IO_HINTS)

    def detect_pure_tensor(self, analysis: HarnessAnalysis, api_name: str) -> bool:
        if not analysis.has_target_call:
            return False
        if analysis.dataset_operations or analysis.resource_variables or analysis.callback_functions:
            return False
        return "random" not in (api_name or "").lower()

    def detect_cpu_parallel(self, api_name: str, analysis: HarnessAnalysis) -> bool:
        if "CPU_PARALLEL_OP" in self.classify_api(api_name):
            return True
        return any("tf.linalg" in c.api_name or "tf.signal" in c.api_name for c in analysis.target_calls)

    # ── helpers ────────────────────────────────────────────────

    def _add(self, scenarios: List[Dict[str, Any]], scenario_id: str, priority: str, reason: str) -> None:
        scenarios.append({
            "scenario_id": scenario_id,
            "priority": priority,
            "reason": reason,
        })

    def _dedupe(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        for s in scenarios:
            cur = best.get(s["scenario_id"])
            if cur is None or rank[s["priority"]] < rank[cur["priority"]]:
                best[s["scenario_id"]] = s
        return list(best.values())

    def _state_model_for(self, scenario_id: str, has_shared_resource: bool) -> str:
        if scenario_id in {"shared_resource_access"} or (has_shared_resource and scenario_id == "independent_api_calls"):
            return "SHARED_MUTABLE"
        if scenario_id == "shared_readonly_input":
            return "SHARED_READONLY"
        if scenario_id in {"tfdata_parallel_map", "tfdata_interleave", "tfdata_early_stop"}:
            return "SHARED_ITERATOR"
        if scenario_id == "python_callback_lifecycle":
            return "THREAD_LOCAL_CONTEXT"
        if scenario_id == "parallel_kernel_execution":
            return "INTERNAL_RUNTIME"
        return "INDEPENDENT_INPUTS"
