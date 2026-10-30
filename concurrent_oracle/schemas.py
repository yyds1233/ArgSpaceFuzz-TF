"""Core data models and status enums for concurrent_oracle.

The JSON shape mirrors section 7 of the design guidance.  These objects are
intentionally plain dataclasses so that every stage can be persisted and
reloaded independently.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import constants


class Decision(str, Enum):
    READY = "READY"
    CONDITIONAL = "CONDITIONAL"
    LOW_PRIORITY = "LOW_PRIORITY"
    INSUFFICIENT_INFO = "INSUFFICIENT_INFO"


class ContractStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNKNOWN = "UNKNOWN"


class ExecutionStatus(str, Enum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    TIMEOUT = "TIMEOUT"
    CRASH = "CRASH"
    SETUP_ERROR = "SETUP_ERROR"
    CAPTURE_ERROR = "CAPTURE_ERROR"


class OracleStatus(str, Enum):
    PASS = "PASS"
    VIOLATION = "VIOLATION"
    INCONCLUSIVE = "INCONCLUSIVE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class GenerationStatus(str, Enum):
    GENERATED = "GENERATED"
    FAILED = "FAILED"
    REPAIRED = "REPAIRED"


class ValidationStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    PENDING = "PENDING"


class StateModel(str, Enum):
    INDEPENDENT_INPUTS = "INDEPENDENT_INPUTS"
    SHARED_READONLY = "SHARED_READONLY"
    SHARED_MUTABLE = "SHARED_MUTABLE"
    SHARED_ITERATOR = "SHARED_ITERATOR"
    SHARED_GENERATOR = "SHARED_GENERATOR"
    THREAD_LOCAL_CONTEXT = "THREAD_LOCAL_CONTEXT"
    INTERNAL_RUNTIME = "INTERNAL_RUNTIME"


# ════════════════════════════════════════════════════════════════
# serialization helpers
# ════════════════════════════════════════════════════════════════

def to_jsonable(value: Any) -> Any:
    """Recursively convert dataclasses/enums/Path to JSON-safe values."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {f.name: to_jsonable(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path | str, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(value), indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _filter_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _coerce_enum(enum_cls, value, default=None):
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(str(value))
    except ValueError:
        return default


# ════════════════════════════════════════════════════════════════
# Harness / analysis
# ════════════════════════════════════════════════════════════════

@dataclass
class HarnessArtifact:
    artifact_id: str
    framework: str = "tensorflow"
    api_name: str = ""
    harness_path: str = ""
    harness_hash: str = ""
    psr_path: Optional[str] = None
    metadata_path: Optional[str] = None
    source: str = "fuzz_api_one"
    tensorflow_version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessArtifact":
        return cls(**_filter_fields(cls, data))


@dataclass
class TargetCall:
    api_name: str
    line: int
    call_expression: str
    is_target: bool = False


@dataclass
class HarnessAnalysis:
    artifact_id: str
    target_calls: List[TargetCall] = field(default_factory=list)
    input_variables: List[str] = field(default_factory=list)
    resource_variables: List[str] = field(default_factory=list)
    callback_functions: List[str] = field(default_factory=list)
    output_expressions: List[str] = field(default_factory=list)
    cleanup_expressions: List[str] = field(default_factory=list)
    ast_parse_status: str = "SUCCESS"
    warnings: List[str] = field(default_factory=list)
    imports: List[Dict[str, Any]] = field(default_factory=list)
    tensor_construction: List[Dict[str, Any]] = field(default_factory=list)
    dataset_operations: List[Dict[str, Any]] = field(default_factory=list)
    resource_lifecycle: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def has_target_call(self) -> bool:
        return any(c.is_target for c in self.target_calls)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HarnessAnalysis":
        kwargs = _filter_fields(cls, data)
        kwargs["target_calls"] = [TargetCall(**_filter_fields(TargetCall, x)) for x in data.get("target_calls", [])]
        return cls(**kwargs)


# ════════════════════════════════════════════════════════════════
# Suitability / plans
# ════════════════════════════════════════════════════════════════

@dataclass
class CandidateScenario:
    scenario_id: str
    oracles: List[str] = field(default_factory=list)
    state_model: str = "INDEPENDENT_INPUTS"
    priority: str = "MEDIUM"
    reason: str = ""


@dataclass
class SuitabilityReport:
    artifact_id: str
    decision: Decision = Decision.INSUFFICIENT_INFO
    api_categories: List[str] = field(default_factory=list)
    rule_candidates: List[str] = field(default_factory=list)
    llm_candidates: List[str] = field(default_factory=list)
    candidate_scenarios: List[CandidateScenario] = field(default_factory=list)
    contract_status: ContractStatus = ContractStatus.UNKNOWN
    required_information: List[str] = field(default_factory=list)
    transformation_difficulty: str = "MEDIUM"
    rationale: str = ""
    confidence: Optional[float] = None
    rule_result: Dict[str, Any] = field(default_factory=dict)
    llm_result: Optional[Dict[str, Any]] = None
    model_config: Dict[str, Any] = field(default_factory=dict)
    prompt_version: Optional[str] = None
    raw_llm_response: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuitabilityReport":
        kwargs = _filter_fields(cls, data)
        kwargs["decision"] = _coerce_enum(Decision, data.get("decision"), Decision.INSUFFICIENT_INFO)
        kwargs["contract_status"] = _coerce_enum(ContractStatus, data.get("contract_status"), ContractStatus.UNKNOWN)
        kwargs["candidate_scenarios"] = [
            CandidateScenario(**_filter_fields(CandidateScenario, x))
            for x in data.get("candidate_scenarios", [])
        ]
        return cls(**kwargs)


@dataclass
class ConcurrentTestPlan:
    plan_id: str
    artifact_id: str
    scenario: str
    runtime: Dict[str, Any] = field(default_factory=dict)
    state_model: str = "INDEPENDENT_INPUTS"
    oracles: List[str] = field(default_factory=list)
    contract_status: ContractStatus = ContractStatus.UNKNOWN
    oracle_contracts: Dict[str, str] = field(default_factory=dict)
    trials: int = 1
    timeout_seconds: int = 60
    case_source: str = ""
    expected_relation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConcurrentTestPlan":
        kwargs = _filter_fields(cls, data)
        kwargs["contract_status"] = _coerce_enum(ContractStatus, data.get("contract_status"), ContractStatus.UNKNOWN)
        return cls(**kwargs)


@dataclass
class ConcurrentHarnessArtifact:
    artifact_id: str
    source_harness: str
    concurrent_harness: str
    plan_id: str
    generation_status: GenerationStatus = GenerationStatus.GENERATED
    validation_status: ValidationStatus = ValidationStatus.PENDING
    generation_attempt: int = 1
    oracle_bindings: List[str] = field(default_factory=list)
    transform_record: Dict[str, Any] = field(default_factory=dict)
    validation_record: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConcurrentHarnessArtifact":
        kwargs = _filter_fields(cls, data)
        kwargs["generation_status"] = _coerce_enum(GenerationStatus, data.get("generation_status"), GenerationStatus.GENERATED)
        kwargs["validation_status"] = _coerce_enum(ValidationStatus, data.get("validation_status"), ValidationStatus.PENDING)
        return cls(**kwargs)


# ════════════════════════════════════════════════════════════════
# Execution / oracle results
# ════════════════════════════════════════════════════════════════

@dataclass
class TaskObservation:
    task_id: int
    input_id: str
    status: str
    output: Any = None
    output_path: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    traceback: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionObservation:
    plan_id: str
    config_id: str
    run_id: Optional[str] = None
    mode: Optional[str] = None
    trials: Optional[int] = None
    baseline_run_id: Optional[str] = None
    test_run_id: Optional[str] = None
    baseline_runtime: Dict[str, Any] = field(default_factory=dict)
    execution_status: ExecutionStatus = ExecutionStatus.COMPLETED
    process_exit_code: Optional[int] = None
    process_signal: Optional[str] = None
    observation_complete: bool = False
    tasks: List[TaskObservation] = field(default_factory=list)
    task_progress: Dict[str, int] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    harness_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    baseline: Any = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionObservation":
        kwargs = _filter_fields(cls, data)
        kwargs["execution_status"] = _coerce_enum(ExecutionStatus, data.get("execution_status"), ExecutionStatus.PARTIAL)
        kwargs["tasks"] = [TaskObservation(**_filter_fields(TaskObservation, x)) for x in data.get("tasks", [])]
        return cls(**kwargs)


@dataclass
class OracleEntry:
    oracle_id: str
    status: OracleStatus
    reason: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OracleResult:
    plan_id: str
    scenario: str
    execution_status: ExecutionStatus
    oracle_results: List[OracleEntry] = field(default_factory=list)
    evidence_level: str = "E1"
    reproduction_rate: Optional[float] = None
    failure_artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OracleResult":
        kwargs = _filter_fields(cls, data)
        kwargs["execution_status"] = _coerce_enum(ExecutionStatus, data.get("execution_status"), ExecutionStatus.PARTIAL)
        kwargs["oracle_results"] = [
            OracleEntry(
                oracle_id=x.get("oracle_id", ""),
                status=_coerce_enum(OracleStatus, x.get("status"), OracleStatus.INCONCLUSIVE),
                reason=x.get("reason", ""),
                evidence=x.get("evidence") or {},
            )
            for x in data.get("oracle_results", [])
        ]
        return cls(**kwargs)


# ════════════════════════════════════════════════════════════════
# id / validation helpers
# ════════════════════════════════════════════════════════════════

def slug(text: str, max_len: int = 80) -> str:
    text = (text or "unknown").strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text[:max_len] or "unknown"


def short_hash(value: str | bytes, n: int = 8) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:n]


def make_artifact_id(api_name: str, harness_hash: str) -> str:
    h = harness_hash.split(":", 1)[-1][:8]
    return f"{slug(api_name)}_{h}"


def make_plan_id(artifact_id: str, scenario: str, index: int = 0) -> str:
    return f"{slug(artifact_id)}__{slug(scenario)}__{index:02d}"


def validate_oracles(oracles: Iterable[str]) -> List[str]:
    bad = [o for o in oracles if o not in constants.ORACLE_IDS]
    if bad:
        raise ValueError(f"unknown oracle ids: {bad}")
    return list(oracles)


def validate_state_model(value: str) -> str:
    if value not in constants.STATE_MODELS:
        raise ValueError(f"unknown state model: {value}")
    return value


def validate_scenario(value: str) -> str:
    if value not in constants.SCENARIO_IDS:
        raise ValueError(f"unknown scenario: {value}")
    return value
