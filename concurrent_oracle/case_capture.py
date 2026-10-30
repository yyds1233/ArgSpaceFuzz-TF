"""Capture fixed, replayable input cases for concurrent tests."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .constants import DEFAULT_INPUT_CASES


_SUPPORTED_ARG_KINDS = {
    "tensor", "ndarray", "sparse_tensor", "ragged_tensor",
    "dtype", "tensor_shape", "python", "list", "tuple", "dict",
}


def _check_arg_desc(desc: Any) -> Dict[str, Any]:
    if not isinstance(desc, dict):
        return {"ok": False, "reason": "argument descriptor is not an object", "conversions": []}
    kind = desc.get("kind")
    if kind not in _SUPPORTED_ARG_KINDS:
        return {"ok": False, "reason": f"unsupported argument kind: {kind}",
                "conversions": []}
    if kind in ("tensor", "ndarray"):
        if desc.get("value") is None or desc.get("value_truncated"):
            return {"ok": False, "reason": f"{kind} argument data is truncated",
                    "conversions": []}
        if not desc.get("dtype") or "shape" not in desc:
            return {"ok": False, "reason": f"{kind} argument lacks dtype/shape",
                    "conversions": []}
        if not desc.get("sha256"):
            return {"ok": False, "reason": f"{kind} argument lacks content digest",
                    "conversions": []}
        conv = ["ndarray_to_tensor"] if kind == "ndarray" else []
        return {"ok": True, "reason": "", "conversions": conv}
    if kind == "dtype":
        return ({"ok": True, "reason": "", "conversions": []}
                if desc.get("name") else
                {"ok": False, "reason": "dtype argument lacks name", "conversions": []})
    if kind == "tensor_shape":
        return ({"ok": True, "reason": "", "conversions": []}
                if isinstance(desc.get("dims"), list) else
                {"ok": False, "reason": "tensor_shape argument lacks dims", "conversions": []})
    if kind == "sparse_tensor":
        for key in ("dtype", "dense_shape", "indices", "values"):
            if key not in desc:
                return {"ok": False, "reason": f"sparse argument lacks {key}", "conversions": []}
        return {"ok": True, "reason": "", "conversions": []}
    if kind == "ragged_tensor":
        fv = desc.get("flat_values")
        if not isinstance(fv, dict):
            return {"ok": False, "reason": "ragged argument lacks flat_values", "conversions": []}
        sub = _check_arg_desc(fv)
        if not sub["ok"]:
            return {"ok": False, "reason": f"ragged flat_values: {sub['reason']}",
                    "conversions": sub["conversions"]}
        if not isinstance(desc.get("nested_row_splits"), list):
            return {"ok": False, "reason": "ragged argument lacks nested_row_splits",
                    "conversions": []}
        return {"ok": True, "reason": "", "conversions": sub["conversions"]}
    if kind in ("list", "tuple", "dict"):
        items = desc.get("items")
        vals = items.values() if isinstance(items, dict) else (items or [])
        conversions = []
        for item in vals:
            sub = _check_arg_desc(item)
            if not sub["ok"]:
                return {"ok": False, "reason": sub["reason"], "conversions": conversions}
            conversions.extend(sub["conversions"])
        return {"ok": True, "reason": "", "conversions": conversions}
    return {"ok": True, "reason": "", "conversions": []}


def check_case_eligibility(case: Dict[str, Any]) -> Dict[str, Any]:
    calls = case.get("target_calls") or []
    if not calls:
        return {"eligible": False, "reason": "no captured target call", "conversions": []}
    conversions = []
    for call in calls:
        if not isinstance(call, dict) or not call.get("call"):
            return {"eligible": False, "reason": "captured call lacks a name", "conversions": conversions}
        for arg in call.get("args") or []:
            sub = _check_arg_desc(arg)
            if not sub["ok"]:
                return {"eligible": False, "reason": sub["reason"], "conversions": conversions}
            conversions.extend(sub["conversions"])
        for arg in (call.get("kwargs") or {}).values():
            sub = _check_arg_desc(arg)
            if not sub["ok"]:
                return {"eligible": False, "reason": sub["reason"], "conversions": conversions}
            conversions.extend(sub["conversions"])
    return {"eligible": True, "reason": "", "conversions": sorted(set(conversions))}


def count_unsupported_target_cases(captured: Dict[str, Dict[str, Any]]) -> int:
    """Count cases that have a captured call but cannot be replayed."""
    count = 0
    for cap in captured.values():
        if cap.get("target_calls") and not check_case_eligibility(cap)["eligible"]:
            count += 1
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
            eligibility = check_case_eligibility(case)
            case["eligible"] = eligibility["eligible"]
            if not eligibility["eligible"]:
                case["ineligible_reason"] = eligibility["reason"]
            if eligibility["conversions"]:
                case["arg_conversions"] = eligibility["conversions"]
        payload["capture_metadata"] = metadata or {}
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return path
