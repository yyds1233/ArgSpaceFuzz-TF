"""Capture fixed, replayable input cases for concurrent tests."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .constants import DEFAULT_INPUT_CASES


def _has_unsupported_arg(value: Any) -> bool:
    if isinstance(value, dict):
        if value.get("kind") == "unsupported_input":
            return True
        return any(_has_unsupported_arg(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_unsupported_arg(v) for v in value)
    return False


def count_unsupported_target_cases(captured: Dict[str, Dict[str, Any]]) -> int:
    count = 0
    for cap in captured.values():
        calls = cap.get("target_calls") or []
        for call in calls:
            if _has_unsupported_arg(call.get("args") or []) or _has_unsupported_arg(call.get("kwargs") or {}):
                count += 1
                break
    return count


class CaseCapture:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)

    def default_cases(self, count: int = 8) -> List[Dict[str, str]]:
        cases = []
        for i, data in enumerate(DEFAULT_INPUT_CASES[:count]):
            cases.append({
                "case_id": f"case_{i:03d}",
                "data_b64": base64.b64encode(data).decode("ascii"),
            })
        return cases

    def write_cases(
        self,
        plan_id: str,
        cases: Optional[Iterable[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        case_dir = self.output_dir / "captured_cases" / plan_id
        case_dir.mkdir(parents=True, exist_ok=True)
        case_list = list(cases) if cases is not None else self.default_cases()
        payload = {
            "plan_id": plan_id,
            "source": (metadata or {}).get("source", "deterministic_defaults"),
            "cases": case_list,
            "metadata": metadata or {},
        }
        path = case_dir / "cases.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_cases(self, path: str | Path) -> List[Dict[str, Any]]:
        p = Path(path)
        if p.is_dir():
            p = p / "cases.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for c in data.get("cases", []):
            item = {"case_id": c.get("case_id")}
            if "data_b64" in c:
                item["data"] = base64.b64decode(c["data_b64"])
            elif "data" in c:
                item["data"] = bytes(c["data"])
            out.append(item)
        return out

    def case_path(self, plan_id: str) -> Path:
        return self.output_dir / "captured_cases" / plan_id / "cases.json"

    def merge_captured_cases(
        self,
        plan_id: str,
        captured: Dict[str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Merge capture-mode target calls into the existing cases.json atomically."""
        path = self.case_path(plan_id)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = {"plan_id": plan_id, "cases": self.default_cases()}
        for case in payload.get("cases", []):
            cap = captured.get(case.get("case_id"))
            if not cap:
                continue
            if cap.get("target_calls"):
                case["target_calls"] = cap["target_calls"]
                case["case_source"] = cap.get("source", "captured_target_call")
            else:
                case["case_source"] = cap.get("source", "bytes_fallback")
                if cap.get("error"):
                    case["capture_error"] = cap["error"]
        payload["capture_metadata"] = metadata or {}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path
