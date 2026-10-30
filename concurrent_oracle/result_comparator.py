"""Compare serialized TensorFlow outputs with structural and numeric tolerance.

The comparator is shared by O2/O3 and the TensorFlow adapter tests.  A result
always carries both ``equal`` and ``inconclusive``; callers must not treat an
inconclusive comparison as a violation or as a pass.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


DEFAULT_COMPARISON = {
    "float_atol": 1e-6,
    "float_rtol": 1e-5,
    "exact_integer": True,
    "compare_nan_mask": True,
    "compare_inf_sign": True,
}

_DESCRIBED_KINDS = {
    "tensor", "ndarray", "sparse_tensor", "ragged_tensor",
    "list", "tuple", "dict", "python", "describe_error",
}


def _cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    c = dict(DEFAULT_COMPARISON)
    if cfg:
        c.update(cfg)
    return c


def _is_described(value: Any) -> bool:
    return isinstance(value, dict) and value.get("kind") in _DESCRIBED_KINDS


def _result(diffs: List[str], inconclusive: bool = False) -> Dict[str, Any]:
    return {
        "equal": not diffs and not inconclusive,
        "diffs": diffs[:50],
        "inconclusive": inconclusive,
    }


def _compare_values(a: Any, b: Any, path: str, cfg: Dict[str, Any], diffs: List[str]) -> bool:
    """Compare arbitrary values.  Returns True if the comparison is inconclusive."""
    if _is_described(a) or _is_described(b):
        r = compare_described(a if isinstance(a, dict) else {}, b if isinstance(b, dict) else {}, cfg)
        diffs.extend(f"{path}: {d}" for d in r.get("diffs", []))
        return bool(r.get("inconclusive"))

    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            diffs.append(f"{path}: keys differ {sorted(a)} vs {sorted(b)}")
            return False
        inc = False
        for k in a:
            inc = _compare_values(a[k], b[k], f"{path}.{k}", cfg, diffs) or inc
        return inc

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
            return False
        inc = False
        for i, (x, y) in enumerate(zip(a, b)):
            inc = _compare_values(x, y, f"{path}[{i}]", cfg, diffs) or inc
        return inc

    if isinstance(a, float) or isinstance(b, float):
        try:
            af, bf = float(a), float(b)
        except Exception:
            diffs.append(f"{path}: non-numeric float comparison ({a!r} vs {b!r})")
            return True
        if math.isnan(af) or math.isnan(bf):
            if cfg.get("compare_nan_mask", True) and math.isnan(af) != math.isnan(bf):
                diffs.append(f"{path}: NaN mask differs")
            return False
        if math.isinf(af) or math.isinf(bf):
            if cfg.get("compare_inf_sign", True) and af != bf:
                diffs.append(f"{path}: inf differs ({af} vs {bf})")
            return False
        atol = float(cfg.get("float_atol", 1e-6))
        rtol = float(cfg.get("float_rtol", 1e-5))
        if abs(af - bf) > atol + rtol * abs(bf):
            diffs.append(f"{path}: float differs ({af} vs {bf})")
        return False

    if isinstance(a, int) and isinstance(b, int):
        if cfg.get("exact_integer", True) and a != b:
            diffs.append(f"{path}: int differs ({a} vs {b})")
        return False

    if a != b:
        diffs.append(f"{path}: value differs ({a!r} vs {b!r})")
    return False


def compare_described(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    c = _cfg(cfg)
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        eq = expected == actual
        return _result([] if eq else ["non-dict output"])
    ek, ak = expected.get("kind"), actual.get("kind")
    if ek == "describe_error" or ak == "describe_error":
        return _result(["output description failed"], inconclusive=True)
    if ek != ak:
        return _result([f"kind differs: {ek} vs {ak}"])
    kind = ek

    if kind in ("tensor", "ndarray"):
        diffs: List[str] = []
        if expected.get("dtype") != actual.get("dtype"):
            diffs.append(f"dtype differs: {expected.get('dtype')} vs {actual.get('dtype')}")
        if expected.get("shape") != actual.get("shape"):
            diffs.append(f"shape differs: {expected.get('shape')} vs {actual.get('shape')}")
        if diffs:
            return _result(diffs)
        if expected.get("sha256") and expected.get("sha256") == actual.get("sha256"):
            return _result([])
        ev, av = expected.get("value"), actual.get("value")
        if ev is not None and av is not None:
            inc = _compare_values(ev, av, "$", c, diffs)
            return _result(diffs, inconclusive=inc)
        return _result(diffs + ["values truncated; hash differs"], inconclusive=True)

    if kind in ("list", "tuple", "dict"):
        diffs = []
        inc = _compare_values(expected.get("items"), actual.get("items"), "$", c, diffs)
        return _result(diffs, inconclusive=inc)

    if kind == "sparse_tensor":
        diffs = []
        inc = False
        for key in ("dtype", "dense_shape", "indices", "values"):
            if expected.get(key) != actual.get(key):
                inc = _compare_values(expected.get(key), actual.get(key), key, c, diffs) or inc
        return _result(diffs, inconclusive=inc)

    if kind == "ragged_tensor":
        diffs = []
        inc = False
        if expected.get("dtype") != actual.get("dtype"):
            diffs.append(f"dtype differs: {expected.get('dtype')} vs {actual.get('dtype')}")
        inc = _compare_values(expected.get("row_splits"), actual.get("row_splits"),
                              "row_splits", c, diffs) or inc
        inc = _compare_values(expected.get("nested_row_splits"),
                              actual.get("nested_row_splits"),
                              "nested_row_splits", c, diffs) or inc
        fv = compare_described(expected.get("flat_values", {}),
                               actual.get("flat_values", {}), c)
        diffs.extend(f"flat_values: {d}" for d in fv.get("diffs", []))
        inc = inc or bool(fv.get("inconclusive"))
        return _result(diffs, inconclusive=inc)

    if kind == "python":
        diffs = []
        inc = _compare_values(expected.get("value"), actual.get("value"), "$", c, diffs)
        return _result(diffs, inconclusive=inc)

    return _result([f"unsupported output kind: {kind}"], inconclusive=True)


def compare_output_lists(
    expected: List[Dict[str, Any]],
    actual: List[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if len(expected) != len(actual):
        return _result([f"output call count differs: {len(expected)} vs {len(actual)}"])
    diffs: List[str] = []
    inconclusive = False
    for i, (e, a) in enumerate(zip(expected, actual)):
        if e.get("call") != a.get("call"):
            diffs.append(f"[{i}] call differs: {e.get('call')} vs {a.get('call')}")
            continue
        if e.get("trial_id") != a.get("trial_id"):
            diffs.append(f"[{i}] trial_id differs: {e.get('trial_id')} vs {a.get('trial_id')}")
        r = compare_described(e.get("output") or {}, a.get("output") or {}, cfg)
        diffs.extend(f"[{i}] {d}" for d in r.get("diffs", []))
        inconclusive = inconclusive or bool(r.get("inconclusive"))
    return _result(diffs, inconclusive=inconclusive)


def compare_task_to_baseline(
    task: Any,
    baseline_entry: Optional[Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if baseline_entry is None:
        return {"equal": False, "diffs": [], "inconclusive": True,
                "reason": "baseline missing"}
    if task.status != "COMPLETED":
        return {"equal": False, "diffs": [], "inconclusive": True,
                "reason": f"task status is {task.status}"}
    expected = baseline_entry.get("outputs") or []
    actual = (task.metadata or {}).get("outputs") or task.output or []
    r = compare_output_lists(expected, actual, cfg)
    r["reason"] = "ok" if r["equal"] else ("; ".join(r["diffs"][:3]) or "inconclusive")
    return r


def find_cross_talk(
    task: Any,
    baseline: Dict[str, Dict[str, Any]],
    cfg: Optional[Dict[str, Any]] = None,
    candidate_input_ids: Optional[List[str]] = None,
    own_baseline: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Return selected other inputs whose baseline matches this task's output.

    Baselines that are indistinguishable from the task's own baseline are
    excluded: a non-injective API can legitimately produce the same output for
    different inputs, which is not evidence of cross-task corruption.
    """
    actual = (task.metadata or {}).get("outputs") or []
    if not actual:
        return []
    own_outputs = (own_baseline or {}).get("outputs") if own_baseline else None
    ids = candidate_input_ids if candidate_input_ids is not None else list(baseline.keys())
    matches: List[str] = []
    for input_id in ids:
        if input_id == task.input_id:
            continue
        expected = (baseline.get(input_id) or {}).get("outputs") or []
        if not expected:
            continue
        if own_outputs:
            own_cmp = compare_output_lists(own_outputs, expected, cfg)
            # If own and other baselines are equal or cannot be distinguished,
            # matching the other baseline is not cross-talk evidence.
            if own_cmp.get("equal") or own_cmp.get("inconclusive"):
                continue
        r = compare_output_lists(expected, actual, cfg)
        if r.get("equal"):
            matches.append(input_id)
    return matches
