"""Configuration loading for concurrent_oracle."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .schemas import to_jsonable


@dataclass
class PathsConfig:
    output_dir: str = "concurrent_oracle/outputs"
    contract_dir: str = "concurrent_oracle/contracts/tensorflow"
    template_dir: str = "concurrent_oracle/templates"


@dataclass
class EnvironmentConfig:
    device: str = "CPU"
    cuda_visible_devices: str = "-1"
    require_cpu_only: bool = True
    multiprocessing_start_method: str = "spawn"


@dataclass
class RuntimeConfig:
    external_threads: List[int] = field(default_factory=lambda: [1, 2, 4])
    intra_op_threads: List[int] = field(default_factory=lambda: [1, 2, 4])
    inter_op_threads: List[int] = field(default_factory=lambda: [1, 2])
    tf_data_parallel_calls: List[int] = field(default_factory=lambda: [1, 2, 4])
    deterministic: List[bool] = field(default_factory=lambda: [True, False])


@dataclass
class GenerationConfig:
    max_repair_attempts: int = 2
    use_template_first: bool = True
    use_llm_transform: bool = True


@dataclass
class ExecutionConfig:
    trials: int = 10
    timeout_seconds: int = 60
    no_progress_timeout_seconds: int = 20
    max_parallel_processes: int = 1


@dataclass
class ComparisonConfig:
    float_atol: float = 1e-6
    float_rtol: float = 1e-5
    exact_integer: bool = True
    compare_nan_mask: bool = True
    compare_inf_sign: bool = True


@dataclass
class SanitizerConfig:
    enabled: bool = False
    type: str = "none"
    log_path: Optional[str] = None


@dataclass
class ModuleConfig:
    framework: str = "tensorflow"
    paths: PathsConfig = field(default_factory=PathsConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    oracle_enabled: List[str] = field(default_factory=lambda: ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8", "O9"])
    comparison: ComparisonConfig = field(default_factory=ComparisonConfig)
    sanitizer: SanitizerConfig = field(default_factory=SanitizerConfig)
    raw: Dict[str, Any] = field(default_factory=dict)

    def output_dir(self) -> Path:
        return Path(self.paths.output_dir)

    def to_dict(self) -> Dict[str, Any]:
        data = to_jsonable(self)
        return data

    def section(self, name: str) -> Dict[str, Any]:
        value = getattr(self, name, None)
        return to_jsonable(value) if value is not None else {}


def _merge_dataclass(obj: Any, data: Dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        return obj
    for f in fields(obj):
        if f.name not in data:
            continue
        value = data[f.name]
        current = getattr(obj, f.name)
        if is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            # YAML may give strings for booleans; preserve explicit types where possible.
            setattr(obj, f.name, value)
    return obj


def default_config() -> ModuleConfig:
    return ModuleConfig()


def load_config(path: Optional[str | Path] = None, overrides: Optional[Dict[str, Any]] = None) -> ModuleConfig:
    cfg = default_config()
    raw: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config not found: {p}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config")
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"config root must be a mapping: {p}")
    if overrides:
        raw = {**raw, **overrides}
    cfg.raw = raw
    _merge_dataclass(cfg, raw)
    return cfg
