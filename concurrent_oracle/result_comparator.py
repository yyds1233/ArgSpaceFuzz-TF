"""Compare serialized TensorFlow outputs with structural and numeric tolerance."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_COMPARISON = {
    "float_atol": 1e-6,
    "float_rtol": 1e-5,
    "exact_integer": True,
    "compare_nan_mask": True,
    "compare_inf_sign": True,
}


def _compare_values(a: Any, b: Any, path: str, cfg: Dict[str, Any], diffs: List[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys differ {sorted(a)} vs {sorted(b)}")
            return
        for k in a:
            _compare_values(a[k], b[k], f"{path}.{k}", cfg, diffs)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _compare_values(x, y, f"{path}[{i}]", cfg, diffs)
        return
    if isinstance(a, float) or isinstance(b, float):
        try:
            af, bf = float(a), float(b)
        except Exception:
            diffs.append(f"{path}: non-numeric float comparison ({a!r} vs {b!r})")
            return
        if math.isnan(af) or math.isnan(bf):
            if cfg.get("compare_nan_mask", True) and math.isnan(af) != math.isnan(bf):
                diffs.append(f"{path}: NaN mask differs")
            return
        if math.isinf(af) or math.isinf(bf):
            if cfg.get("compare_inf_sign", True) and af != bf:
                diffs.append(f"{path}: inf differs ({af} vs {bf})")
            return
        atol = float(cfg.get("float_atol", 1e-6))
        rtol = float(cfg.get("float_rtol", 1e-5))
        if abs(af - bf) > atol + rtol * abs(bf):
            diffs.append(f"{path}: float differs ({af} vs {bf})")
        return
    if isinstance(a, int) and isinstance(b, int):
        if cfg.get("exact_integer", True) and a != b:
            diffs.append(f"{path}: int differs ({a} vs {b})")
        return
    if a != b:
        diffs.append(f"{path}: value differs ({a!r} vs {b!r})")


def compare_described(expected: Dict[str, Any], actual: Dict[str, Any],
                      cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    c = dict(DEFAULT_COMPARISON)
    if cfg:
        c.update(cfg)
    diffs: List[str] = []
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return {"equal": expected == actual, "diffs": [] if expected == actual else ["non-dict output"],
                "inconclusive": False}
    if expected.get("kind") != actual.get("kind"):
        return {"equal": False, "diffs": [f"kind differs: {expected.get('kind')} vs {actual.get('kind')}"],
                "inconclusive": False}
    kind = expected.get("kind")
    if kind in ("tensor", "ndarray"):
        if expected.get("dtype") != actual.get("dtype"):
            diffs.append(f"dtype differs: {expected.get('dtype')} vs {actual.get('dtype')}")
        if expected.get("shape") != actual.get("shape"):
            diffs.append(f"shape differs: {expected.get('shape')} vs {actual.get('shape')}")
        if diffs:
            return {"equal": False, "diffs": diffs[:50], "inconclusive": False}
        if expected.get("sha256") and expected.get("sha256") == actual.get("sha256"):
            return {"equal": True, "diffs": [], "inconclusive": False}
        ev, av = expected.get("value"), actual.get("value")
        if ev is not None and av is not None:
            _compare_values(ev, av, "$", c, diffs)
            return {"equal": not diffs, "diffs": diffs[:50], "inconclusive": False}
        # Hashes differ but full values are unavailable: cannot distinguish a
        # legal floating-point order difference from a real mismatch.
        return {"equal": False, "diffs": diffs + ["values truncated; hash differs"],
                "inconclusive": True}
    if kind in ("list", "tuple", "dict"):
        _compare_values(expected.get("items"), actual.get("items"), "$", c, diffs)
        return {"equal": not diffs, "diffs": diffs[:50], "inconclusive": False}
    if kind == "sparse_tensor":
        for key in ("dense_shape", "indices", "values"):
            if expected.get(key) != actual.get(key):
                _compare_values(expected.get(key), actual.get(key), key, c, diffs)
        return {"equal": not diffs, "diffs": diffs[:50], "inconclusive": False}
    if kind == "ragged_tensor":
        _compare_values(expected.get("row_splits"), actual.get("row_splits"), "row_splits", c, diffs)
        fv = compare_described(expected.get("flat_values", {}), actual.get("flat_values", {}), cfg)
        diffs.extend(fv.get("diffs", []))
        return {"equal": not diffs, "diffs": diffs[:50], "inconclusive": fv.get("inconclusive", False)}
    if kind == "python":
        _compare_values(expected.get("value"), actual.get("value"), "$", c, diffs)
        return {"equal": not diffs, "diffs": diffs[:50], "inconclusive": False}
    return {"equal": expected == actual, "diffs": [] if expected == actual else ["unknown output kind"],
            "inconclusive": False}


def compare_output_lists(expected: List[Dict[str, Any]], actual: List[Dict[str, Any]],
                         cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if len(expected) != len(actual):
        return {"equal": False, "diffs": [f"output call count differs: {len(expected)} vs {len(actual)}"],
                "inconclusive": False}
    diffs: List[str] = []
    inconclusive = False
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e.get("call") != a.get("call"):
            diffs.append(f"[{i}] call differs: {e.get('call')} vs {a.get('call')}")
            continue
        r = compare_described(e.get("output") or {}, a.get("output") or {}, cfg)
        diffs.extend(f"[{i}] {d}" for d in r.get("diffs", []))
        inconclusive = inconclusive or bool(r.get("inconclusive"))
    return {"equal": not diffs, "diffs": diffs[:50], "inconclusive": inconclusive}


def compare_task_to_baseline(task: Any, baseline_entry: Optional[Dict[str, Any]],
                             cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if baseline_entry is None:
        return {"equal": False, "diffs": [], "inconclusive": True, "reason": "baseline missing"}
    if task.status != "COMPLETED":
        return {"equal": False, "diffs": [], "inconclusive": True,
                "reason": f"task status is {task.status}"}
    expected = baseline_entry.get("outputs") or []
    actual = (task.metadata or {}).get("outputs") or task.output or []
    r = compare_output_lists(expected, actual, cfg)
    r["reason"] = "ok" if r["equal"] else "; ".join(r["diffs"][:3])
    return r


def find_cross_talk(task: Any, baseline: Dict[str, Dict[str, Any]],
                    cfg: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return other input_ids whose baseline matches this task's outputs."""
    actual = (task.metadata or {}).get("outputs") or []
    if not actual:
        return []
    matches: List[str] = []
    for input_id, entry in baseline.items():
        if input_id == task.input_id:
            continue
        expected = (entry or {}).get("outputs") or []
        if not expected:
            continue
        r = compare_output_lists(expected, actual, cfg)
        if r.get("equal"):
            matches.append(input_id)
    return matches
