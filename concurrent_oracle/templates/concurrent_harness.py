#!/usr/bin/env python3
"""Generated concurrent harness template for one TensorFlow API.

Placeholders replaced by concurrent_oracle.transformer:
  ORIGINAL_HARNESS = __ORIGINAL_HARNESS__
  TARGET_PATCHES   = __TARGET_PATCHES__
  DEFAULT_CASES    = __DEFAULT_CASES__

Modes:
  capture     : run the original TestOneInput once per case and capture the
                target call's real args/kwargs plus baseline output.
  baseline    : replay captured target calls, or fall back to TestOneInput,
                serially; no worker pool is created.
  concurrent  : replay captured target calls (or TestOneInput) in N workers;
                no baseline is run in this process.
  both        : legacy convenience mode (baseline + concurrent).
"""
from __future__ import annotations

import base64
import faulthandler
import hashlib
import importlib.util
import json
import os
import signal
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

ORIGINAL_HARNESS = __ORIGINAL_HARNESS__
TARGET_PATCHES = __TARGET_PATCHES__
DEFAULT_CASES = __DEFAULT_CASES__

_PATCH_STATE = threading.local()
_EVENT_LOCK = threading.Lock()
_EVENTS_PATH: Optional[str] = None
_DETERMINISM_STATE: Dict[str, Any] = {}

# The parent sends SIGUSR1 before killing a stalled process; the stack dump
# goes to stderr, which the runner captures in stderr.log.
faulthandler.enable()
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass


# ── events / serialization ─────────────────────────────────────

def _event(name: str, **fields: Any) -> None:
    path = _EVENTS_PATH
    if not path:
        return
    rec = {"event": name, "ts": time.monotonic()}
    rec.update(fields)
    try:
        with _EVENT_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if np is not None and isinstance(value, np.generic):
        return _jsonable(value.item())
    if np is not None and isinstance(value, np.ndarray):
        # 0-d arrays return a Python scalar from tolist(); recurse instead of
        # iterating so scalar tensors remain serializable.
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    return value


def _array_sha256(arr) -> str:
    import hashlib
    import json
    if arr.dtype.kind in ("O", "U", "S"):
        payload = json.dumps(_jsonable(arr.tolist()), ensure_ascii=False,
                             sort_keys=True).encode("utf-8")
    else:
        payload = arr.tobytes()
    return hashlib.sha256(payload).hexdigest()


def _describe(value: Any) -> Dict[str, Any]:
    import numpy as np
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        arr = value.numpy()
        desc = {
            "kind": "tensor",
            "dtype": value.dtype.name,
            "shape": [int(x) for x in value.shape],
            "sha256": _array_sha256(arr),
        }
        if int(arr.size) <= 65536:
            desc["value"] = _jsonable(arr)
        else:
            desc["value_truncated"] = True
        return desc

    if isinstance(value, tf.SparseTensor):
        return {
            "kind": "sparse_tensor",
            "dtype": value.dtype.name,
            "dense_shape": [int(x) for x in value.dense_shape.numpy().tolist()],
            "indices": _jsonable(value.indices.numpy()),
            "values": _jsonable(value.values.numpy()),
        }

    if isinstance(value, tf.RaggedTensor):
        return {
            "kind": "ragged_tensor",
            "dtype": value.dtype.name,
            "flat_values": _describe(value.flat_values),
            "row_splits": _jsonable(value.row_splits.numpy()),
            "nested_row_splits": [_jsonable(x.numpy()) for x in value.nested_row_splits],
        }

    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
            "value": _jsonable(value) if value.size <= 65536 else None,
        }

    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "items": [_safe_describe(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(k): _safe_describe(v) for k, v in value.items()}}
    return {"kind": "python", "type": type(value).__name__, "value": _jsonable(value)}


def _safe_describe(value: Any) -> Dict[str, Any]:
    try:
        return _describe(value)
    except Exception as exc:
        return {"kind": "describe_error", "error_type": type(exc).__name__, "error": str(exc)}


def _describe_input(value: Any) -> Dict[str, Any]:
    """Describe a target-call argument for later replay."""
    import numpy as np
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        arr = value.numpy()
        desc = {
            "kind": "tensor",
            "dtype": value.dtype.name,
            "shape": [int(x) for x in value.shape],
            "sha256": _array_sha256(arr),
        }
        if int(arr.size) <= 65536:
            desc["value"] = _jsonable(arr)
        else:
            desc["value_truncated"] = True
        return desc
    if isinstance(value, tf.DType):
        return {"kind": "dtype", "name": value.name}
    if isinstance(value, tf.TensorShape):
        return {"kind": "tensor_shape", "dims": list(value.as_list())}
    if isinstance(value, tf.SparseTensor):
        return {
            "kind": "sparse_tensor",
            "dtype": value.dtype.name,
            "dense_shape": [int(x) for x in value.dense_shape.numpy().tolist()],
            "indices": _jsonable(value.indices.numpy()),
            "values": _jsonable(value.values.numpy()),
        }
    if isinstance(value, tf.RaggedTensor):
        return {
            "kind": "ragged_tensor",
            "dtype": value.dtype.name,
            "flat_values": _describe_input(value.flat_values),
            "nested_row_splits": [_jsonable(x.numpy()) for x in value.nested_row_splits],
        }
    if isinstance(value, np.generic):
        return {"kind": "python", "value": _jsonable(value.item())}
    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": _array_sha256(value),
            "value": _jsonable(value) if value.size <= 65536 else None,
        }
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__,
                "items": [_describe_input(v) for v in value]}
    if isinstance(value, dict):
        return {"kind": "dict", "items": {str(k): _describe_input(v) for k, v in value.items()}}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "python", "value": _jsonable(value)}
    return {"kind": "unsupported_input", "type": type(value).__name__}


def _from_jsonable(value: Any, dtype: Any = None) -> Any:
    if isinstance(value, dict) and set(value.keys()) == {"__bytes_b64__"}:
        return base64.b64decode(value["__bytes_b64__"])
    if isinstance(value, dict) and set(value.keys()) == {"real", "imag"}:
        return complex(value["real"], value["imag"])
    if isinstance(value, list):
        return [_from_jsonable(v, dtype) for v in value]
    if isinstance(value, dict):
        return {k: _from_jsonable(v, dtype) for k, v in value.items()}
    return value


def _reconstruct_arg(desc: Dict[str, Any], tf):
    kind = desc.get("kind")
    if kind in ("tensor", "ndarray"):
        value = desc.get("value")
        if value is None:
            raise ValueError("captured argument is truncated and cannot be replayed")
        dtype = desc.get("dtype")
        shape = desc.get("shape")
        tensor = tf.constant(_from_jsonable(value, dtype), dtype=dtype, shape=shape)
        if desc.get("sha256"):
            actual = _array_sha256(tensor.numpy())
            if actual != desc["sha256"]:
                raise ValueError("reconstructed argument digest mismatch")
        return tensor
    if kind == "dtype":
        return tf.dtypes.as_dtype(desc.get("name"))
    if kind == "tensor_shape":
        return tf.TensorShape(desc.get("dims"))
    if kind == "sparse_tensor":
        return tf.sparse.SparseTensor(
            indices=tf.constant(desc.get("indices"), dtype=tf.int64),
            values=tf.constant(_from_jsonable(desc.get("values")), dtype=desc.get("dtype")),
            dense_shape=tf.constant(desc.get("dense_shape"), dtype=tf.int64),
        )
    if kind == "ragged_tensor":
        flat = _reconstruct_arg(desc.get("flat_values") or {}, tf)
        splits = [tf.constant(x, dtype=tf.int64) for x in (desc.get("nested_row_splits") or [])]
        return tf.RaggedTensor.from_nested_row_splits(flat, splits)
    if kind in ("list", "tuple", "dict"):
        items = desc.get("items")
        if kind == "dict":
            return {k: _reconstruct_arg(v, tf) for k, v in (items or {}).items()}
        vals = [_reconstruct_arg(v, tf) for v in (items or [])]
        return tuple(vals) if kind == "tuple" else vals
    if kind == "python":
        return _from_jsonable(desc.get("value"))
    raise ValueError(f"unsupported captured argument kind: {kind}")


# ── target patching ────────────────────────────────────────────

def _record() -> Optional[Dict[str, Any]]:
    return getattr(_PATCH_STATE, "record", None)


def _patch_targets(tf) -> Dict[str, Any]:
    originals: Dict[str, Any] = {}
    for dotted in TARGET_PATCHES:
        parts = dotted.split(".")
        if len(parts) < 2 or parts[0] != "tf":
            continue
        obj = tf
        ok = True
        for part in parts[1:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        attr = parts[-1]
        if not ok or not hasattr(obj, attr):
            continue
        original = getattr(obj, attr)

        def make_wrapper(orig, call_name):
            def wrapper(*args, **kwargs):
                rec = _record()
                call_index = None
                if rec is not None:
                    rec["call_index"] = int(rec.get("call_index", 0)) + 1
                    call_index = rec["call_index"]
                    _event("TARGET_ENTER", call=call_name, task_id=rec.get("task_id"),
                           trial_id=rec.get("trial_id"), call_index=call_index)
                    if rec.get("capture_args"):
                        try:
                            rec["calls"].append({
                                "call": call_name,
                                "args": [_describe_input(a) for a in args],
                                "kwargs": {k: _describe_input(v) for k, v in kwargs.items()},
                            })
                        except Exception as exc:
                            rec["capture_error"] = f"{type(exc).__name__}: {exc}"
                try:
                    out = orig(*args, **kwargs)
                except Exception as exc:
                    if rec is not None:
                        rec["exceptions"].append({
                            "call": call_name, "trial_id": rec.get("trial_id"),
                            "call_index": call_index,
                            "type": type(exc).__name__, "message": str(exc),
                        })
                    _event("TARGET_EXCEPTION", call=call_name, task_id=rec.get("task_id") if rec else None,
                           trial_id=rec.get("trial_id") if rec else None,
                           type=type(exc).__name__)
                    raise
                if rec is not None:
                    _event("TARGET_RETURN", call=call_name, task_id=rec.get("task_id"),
                           trial_id=rec.get("trial_id"), call_index=call_index)
                    _event("DESCRIBE_START", call=call_name, task_id=rec.get("task_id"),
                           trial_id=rec.get("trial_id"), call_index=call_index)
                    described = _safe_describe(out)
                    _event("DESCRIBE_END", call=call_name, task_id=rec.get("task_id"),
                           trial_id=rec.get("trial_id"), call_index=call_index)
                    rec["outputs"].append({
                        "call": call_name, "trial_id": rec.get("trial_id"),
                        "call_index": call_index,
                        "output": described,
                    })
                return out
            return wrapper

        setattr(obj, attr, make_wrapper(original, dotted))
        originals[dotted] = (obj, attr, original)
    return originals


def _unpatch_targets(originals: Dict[str, Any]) -> None:
    for _, (obj, attr, original) in originals.items():
        try:
            setattr(obj, attr, original)
        except Exception:
            pass


# ── runtime / cases ────────────────────────────────────────────

def _setup_tf(runtime: Dict[str, Any], mode: str = "concurrent", seed: int = 0):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    intra = int(runtime.get("intra_op_threads", 1))
    inter = int(runtime.get("inter_op_threads", 1))
    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)
    det = runtime.get("deterministic")
    _DETERMINISM_STATE.clear()
    _DETERMINISM_STATE.update({
        "deterministic_requested": det,
        "deterministic_applied": None,
        "deterministic_error": None,
        "deterministic_method": None,
        "seed": None,
    })
    if det is not None:
        try:
            if bool(det):
                tf.config.experimental.enable_op_determinism()
                _DETERMINISM_STATE["deterministic_method"] = "enable_op_determinism"
            elif hasattr(tf.config.experimental, "disable_op_determinism"):
                tf.config.experimental.disable_op_determinism()
                _DETERMINISM_STATE["deterministic_method"] = "disable_op_determinism"
            else:
                # A fresh process starts with op determinism disabled.
                _DETERMINISM_STATE["deterministic_method"] = "default_disabled"
            _DETERMINISM_STATE["deterministic_applied"] = True
        except Exception as exc:
            _DETERMINISM_STATE["deterministic_applied"] = False
            _DETERMINISM_STATE["deterministic_error"] = f"{type(exc).__name__}: {exc}"
    if mode == "capture":
        # Freeze random construction in the single-threaded capture phase.
        try:
            tf.random.set_seed(int(seed))
            _DETERMINISM_STATE["seed"] = int(seed)
        except Exception as exc:
            _DETERMINISM_STATE["deterministic_error"] = (
                f"seed setup failed: {type(exc).__name__}: {exc}"
            )
    gpus = tf.config.list_physical_devices("GPU")
    if runtime.get("require_cpu_only", True) and gpus:
        raise RuntimeError(f"CPU-only test expected, but GPU devices are visible: {gpus}")
    return tf


def _load_cases(case_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = case_config.get("cases_path")
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cases = data.get("cases", [])
        out = []
        for i, c in enumerate(cases):
            item = {"case_id": c.get("case_id", f"case_{i:03d}")}
            if "data_b64" in c:
                item["data_b64"] = c["data_b64"]
            elif "data" in c:
                item["data_b64"] = base64.b64encode(bytes(c["data"])).decode("ascii")
            if c.get("target_calls"):
                item["target_calls"] = c["target_calls"]
            if "eligible" in c:
                item["eligible"] = bool(c["eligible"])
            if c.get("ineligible_reason"):
                item["ineligible_reason"] = c["ineligible_reason"]
            if c.get("arg_conversions"):
                item["arg_conversions"] = c["arg_conversions"]
            if c.get("case_source"):
                item["case_source"] = c["case_source"]
            out.append(item)
        if out:
            flagged = [c for c in out if "eligible" in c]
            if flagged:
                eligible = [c for c in out if c.get("eligible")]
                skipped = [c for c in out if not c.get("eligible")]
                if skipped:
                    _event("SKIPPED_INELIGIBLE_CASES", count=len(skipped),
                           reasons=[c.get("ineligible_reason") for c in skipped[:20]])
                if not eligible:
                    raise RuntimeError("no eligible captured cases can be replayed")
                out = eligible
            return out
    return [{"case_id": f"case_{i:03d}", "data_b64": x} for i, x in enumerate(DEFAULT_CASES)]


def _decode(data_b64: str) -> bytes:
    return base64.b64decode(data_b64.encode("ascii"))


# ── execution ──────────────────────────────────────────────────

def _new_record(task_id: int, input_id: str, trial_id: int, capture_args: bool = False) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "input_id": input_id,
        "trial_id": trial_id,
        "status": "RUNNING",
        "outputs": [],
        "exceptions": [],
        "calls": [],
        "capture_args": capture_args,
        "call_index": 0,
        "expected_calls": None,
        "actual_calls": 0,
        "complete": False,
        "started_at": time.time(),
        "finished_at": None,
    }


def _build_shared_calls(case: Dict[str, Any], originals: Dict[str, Any], tf) -> List[Any]:
    calls = []
    for spec in case.get("target_calls") or []:
        dotted = spec.get("call")
        if dotted not in originals:
            raise RuntimeError(f"no original function for captured call {dotted}")
        original = originals[dotted][2]
        args = [_reconstruct_arg(a, tf) for a in spec.get("args", [])]
        kwargs = {k: _reconstruct_arg(v, tf) for k, v in (spec.get("kwargs") or {}).items()}
        calls.append((dotted, original, args, kwargs))
    return calls


def _invoke_once(module, case: Dict[str, Any], task_id: int, trial_id: int,
                 originals: Dict[str, Any], tf, capture_args: bool = False,
                 shared_calls: Optional[List[Any]] = None,
                 expected_calls: Optional[int] = None) -> Dict[str, Any]:
    rec = _new_record(task_id, case["case_id"], trial_id, capture_args)
    rec["expected_calls"] = expected_calls
    _PATCH_STATE.record = rec
    try:
        if shared_calls is not None:
            # Shared-readonly backend: every worker uses the same argument
            # objects constructed once by the parent thread.
            for dotted, original, args, kwargs in shared_calls:
                rec["call_index"] = int(rec.get("call_index", 0)) + 1
                call_index = rec["call_index"]
                _event("TARGET_ENTER", call=dotted, task_id=task_id,
                       trial_id=trial_id, call_index=call_index)
                try:
                    out = original(*args, **kwargs)
                    _event("TARGET_RETURN", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    _event("DESCRIBE_START", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    described = _safe_describe(out)
                    _event("DESCRIBE_END", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    rec["outputs"].append({
                        "call": dotted, "trial_id": trial_id, "call_index": call_index,
                        "output": described,
                    })
                except Exception as exc:
                    rec["exceptions"].append({
                        "call": dotted, "trial_id": trial_id, "call_index": call_index,
                        "type": type(exc).__name__, "message": str(exc),
                    })
                    _event("TARGET_EXCEPTION", call=dotted, task_id=task_id,
                           trial_id=trial_id, type=type(exc).__name__)
                    raise
        elif case.get("target_calls") and originals:
            # Replay the captured target call arguments directly.  This freezes
            # the real inputs even when TestOneInput uses random sources.
            for spec in case["target_calls"]:
                dotted = spec.get("call")
                if dotted not in originals:
                    raise RuntimeError(f"no original function for captured call {dotted}")
                original = originals[dotted][2]
                rec["call_index"] = int(rec.get("call_index", 0)) + 1
                call_index = rec["call_index"]
                try:
                    _event("ARG_REBUILD_START", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    args = [_reconstruct_arg(a, tf) for a in spec.get("args", [])]
                    kwargs = {k: _reconstruct_arg(v, tf) for k, v in (spec.get("kwargs") or {}).items()}
                    _event("ARG_REBUILD_END", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    _event("TARGET_ENTER", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    out = original(*args, **kwargs)
                    _event("TARGET_RETURN", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    _event("DESCRIBE_START", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    described = _safe_describe(out)
                    _event("DESCRIBE_END", call=dotted, task_id=task_id,
                           trial_id=trial_id, call_index=call_index)
                    rec["outputs"].append({
                        "call": dotted, "trial_id": trial_id, "call_index": call_index,
                        "output": described,
                    })
                except Exception as exc:
                    rec["exceptions"].append({
                        "call": dotted, "trial_id": trial_id, "call_index": call_index,
                        "type": type(exc).__name__, "message": str(exc),
                    })
                    _event("TARGET_EXCEPTION", call=dotted, task_id=task_id,
                           trial_id=trial_id, type=type(exc).__name__)
                    raise
        else:
            module.TestOneInput(_decode(case["data_b64"]))
        if rec["exceptions"] and not rec["outputs"]:
            rec["status"] = "EXCEPTION"
        elif not rec["outputs"]:
            rec["status"] = "NO_TARGET_CALL"
        else:
            rec["status"] = "COMPLETED"
    except Exception as exc:
        rec["status"] = "EXCEPTION"
        rec["exceptions"].append({
            "call": None, "trial_id": trial_id,
            "type": type(exc).__name__, "message": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        rec["finished_at"] = time.time()
        rec["actual_calls"] = len(rec.get("outputs") or []) + len(rec.get("exceptions") or [])
        rec["complete"] = bool(
            not rec.get("exceptions")
            and rec.get("status") == "COMPLETED"
            and (expected_calls is None or len(rec.get("outputs") or []) == expected_calls)
        )
        rec.pop("capture_args", None)
        _PATCH_STATE.record = None
    return rec


def _summarize_trial_status(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    complete_trials = sum(1 for r in records if r.get("complete"))
    failed_trials = sum(1 for r in records if r.get("status") == "EXCEPTION")
    no_target_trials = sum(1 for r in records if r.get("status") == "NO_TARGET_CALL")
    if total > 0 and complete_trials == total:
        status = "COMPLETED"
    elif complete_trials > 0:
        status = "PARTIAL"
    elif failed_trials > 0:
        status = "EXCEPTION"
    else:
        status = "NO_TARGET_CALL"
    return {
        "status": status,
        "complete": total > 0 and complete_trials == total,
        "trials_expected": total,
        "trials_completed": complete_trials,
        "trials_failed": failed_trials,
        "trials_no_target": no_target_trials,
        "trial_status_counts": {
            "COMPLETED": sum(1 for r in records if r.get("status") == "COMPLETED"),
            "EXCEPTION": failed_trials,
            "NO_TARGET_CALL": no_target_trials,
        },
    }


def _invoke(module, case: Dict[str, Any], task_id: int, trials: int,
            originals: Dict[str, Any], tf, capture_args: bool = False,
            shared_calls: Optional[List[Any]] = None) -> Dict[str, Any]:
    if shared_calls is not None:
        expected_calls = len(shared_calls)
    elif case.get("target_calls"):
        expected_calls = len(case.get("target_calls") or [])
    else:
        expected_calls = None
    records = [
        _invoke_once(module, case, task_id, trial_id, originals, tf, capture_args,
                     shared_calls, expected_calls)
        for trial_id in range(trials)
    ]
    outputs: List[Dict[str, Any]] = []
    exceptions: List[Dict[str, Any]] = []
    for r in records:
        outputs.extend(r.get("outputs") or [])
        exceptions.extend(r.get("exceptions") or [])
    summary = _summarize_trial_status(records)
    calls: List[Dict[str, Any]] = []
    for r in records:
        calls.extend(r.get("calls") or [])
    return {
        "task_id": task_id,
        "input_id": case["case_id"],
        "status": summary["status"],
        "complete": summary["complete"],
        "trials": trials,
        "trials_expected": summary["trials_expected"],
        "trials_completed": summary["trials_completed"],
        "trials_failed": summary["trials_failed"],
        "trial_status_counts": summary["trial_status_counts"],
        "expected_calls": (expected_calls * trials) if expected_calls is not None else None,
        "actual_calls": len(outputs) + len(exceptions),
        "arg_conversions": case.get("arg_conversions") or [],
        "trial_records": records,
        "outputs": outputs,
        "exceptions": exceptions,
        "calls": calls,
        "started_at": min((r.get("started_at") or time.time()) for r in records),
        "finished_at": max((r.get("finished_at") or time.time()) for r in records),
    }


def _worker(module, task_id: int, case: Dict[str, Any], barrier, results, lock,
            trials: int, originals, tf, shared_calls=None):
    rec = {
        "task_id": task_id,
        "input_id": case["case_id"],
        "status": "PENDING",
        "complete": False,
        "trials": trials,
        "outputs": [],
        "exceptions": [],
        "started_at": None,
        "finished_at": None,
    }
    try:
        _event("BARRIER_WAIT", task_id=task_id, input_id=case["case_id"])
        barrier.wait(timeout=30)
        _event("BARRIER_RELEASED", task_id=task_id, input_id=case["case_id"])
        _event("TASK_STARTED", task_id=task_id, input_id=case["case_id"])
        rec["started_at"] = time.time()
        item = _invoke(module, case, task_id, trials, originals, tf,
                       shared_calls=shared_calls)
        rec.update(item)
    except threading.BrokenBarrierError:
        rec["status"] = "BARRIER_ERROR"
        rec["exceptions"].append({"type": "BrokenBarrierError", "message": "barrier timed out"})
    except Exception as exc:
        rec["status"] = "EXCEPTION"
        rec["exceptions"].append({"type": type(exc).__name__, "message": str(exc),
                                  "traceback": traceback.format_exc()})
    finally:
        rec["finished_at"] = time.time()
        _event("TASK_FINISHED", task_id=task_id, status=rec.get("status"))
        with lock:
            results.append(rec)


def run_case(case_config: Dict[str, Any]) -> Dict[str, Any]:
    global _EVENTS_PATH
    _EVENTS_PATH = case_config.get("events_path")
    mode = case_config.get("mode", "both")
    if case_config.get("capture_only"):
        mode = "capture"
    trials = max(1, int(case_config.get("trials", 1)))
    runtime = dict(case_config.get("runtime") or {})
    seed = int(case_config.get("seed", 0))
    tf = _setup_tf(runtime, mode, seed)
    _event("FRAMEWORK_READY", tf_version=getattr(tf, "__version__", None),
           pid=os.getpid(), mode=mode, trials=trials)

    spec = importlib.util.spec_from_file_location(
        "concurrent_original_harness", ORIGINAL_HARNESS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original harness: {ORIGINAL_HARNESS}")
    originals = _patch_targets(tf)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cases = _load_cases(case_config)
        _event("CASES_READY", count=len(cases),
               captured=sum(1 for c in cases if c.get("target_calls")))
        baseline: Dict[str, Any] = {}
        if mode in ("capture", "baseline", "both"):
            for i, case in enumerate(cases):
                baseline[case["case_id"]] = _invoke(
                    module, case, i, 1 if mode == "capture" else trials,
                    originals, tf, capture_args=(mode == "capture"),
                )
        baseline_complete = bool(baseline) and all(
            v.get("complete") for v in baseline.values()
        )

        if mode == "capture":
            captured = {}
            for case in cases:
                rec = baseline.get(case["case_id"]) or {}
                calls = rec.get("calls") or []
                if calls:
                    captured[case["case_id"]] = {"target_calls": calls,
                                                  "source": "captured_target_call"}
                elif rec.get("capture_error"):
                    captured[case["case_id"]] = {"target_calls": [],
                                                  "source": "capture_error",
                                                  "error": rec["capture_error"]}
                else:
                    captured[case["case_id"]] = {"target_calls": [],
                                                  "source": "no_target_call"}
            return {
                "run_id": case_config.get("run_id"),
                "plan_id": case_config.get("plan_id"),
                "config_id": case_config.get("config_id"),
                "mode": mode,
                "trials": 1,
                "execution_status": "COMPLETED",
                "observation_complete": True,
                "runtime": _runtime_info(tf, runtime),
                "baseline": baseline,
                "captured_cases": captured,
                "tasks": [],
                "task_progress": {"submitted": 0, "started": 0, "completed": 0,
                                  "failed": 0, "pending": 0},
                "config": case_config,
            }

        if mode == "baseline":
            return {
                "run_id": case_config.get("run_id"),
                "plan_id": case_config.get("plan_id"),
                "config_id": case_config.get("config_id"),
                "mode": mode,
                "trials": trials,
                "execution_status": "COMPLETED",
                "observation_complete": baseline_complete,
                "baseline_complete": baseline_complete,
                "runtime": _runtime_info(tf, runtime),
                "baseline": baseline,
                "tasks": [],
                "task_progress": {"submitted": 0, "started": 0, "completed": 0,
                                  "failed": 0, "pending": 0},
                "config": case_config,
            }

        n = max(1, int(runtime.get("external_threads", 2)))
        state_model = case_config.get("state_model", "INDEPENDENT_INPUTS")
        shared_calls = None
        shared_input_id = None
        if state_model == "SHARED_READONLY":
            shared_case = next((c for c in cases if c.get("target_calls")), cases[0])
            shared_input_id = shared_case["case_id"]
            if shared_case.get("target_calls"):
                shared_calls = _build_shared_calls(shared_case, originals, tf)
                _event("SHARED_ARGS_READY", count=len(shared_calls),
                       input_id=shared_input_id)
            selected = [shared_case] * n
        else:
            selected = [cases[i % len(cases)] for i in range(n)]
        barrier = threading.Barrier(n)
        results: List[Dict[str, Any]] = []
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(_worker, module, task_id, case, barrier, results, lock,
                                trials, originals, tf, shared_calls)
                for task_id, case in enumerate(selected)
            ]
            for f in futures:
                f.result()

        results.sort(key=lambda x: x["task_id"])
        completed = sum(1 for r in results if r.get("status") == "COMPLETED")
        partial = sum(1 for r in results if r.get("status") == "PARTIAL")
        failed = sum(1 for r in results if r.get("status") in ("EXCEPTION", "BARRIER_ERROR"))
        progress = {
            "submitted": n,
            "started": sum(1 for r in results if r.get("started_at") is not None),
            "completed": completed,
            "partial": partial,
            "failed": failed,
            "pending": max(0, n - len(results)),
        }
        observation_complete = (
            len(results) == n
            and all(r.get("status") == "COMPLETED" and r.get("complete") for r in results)
        )
        return {
            "run_id": case_config.get("run_id"),
            "plan_id": case_config.get("plan_id"),
            "config_id": case_config.get("config_id"),
            "mode": mode,
            "trials": trials,
            "execution_status": "COMPLETED",
            "observation_complete": observation_complete,
            "baseline_complete": baseline_complete if mode == "both" else None,
            "runtime": _runtime_info(tf, runtime),
            "baseline": baseline if mode == "both" else {},
            "shared_input_id": shared_input_id,
            "tasks": results,
            "task_progress": progress,
            "config": case_config,
        }
    finally:
        _unpatch_targets(originals)


def _runtime_info(tf, runtime: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tensorflow_version": getattr(tf, "__version__", None),
        "device": "CPU",
        "gpu_devices": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "intra_op_threads": int(runtime.get("intra_op_threads", 1)),
        "inter_op_threads": int(runtime.get("inter_op_threads", 1)),
        "external_threads": int(runtime.get("external_threads", 1)),
        "tf_data_parallel_calls": int(runtime.get("tf_data_parallel_calls", 1)),
        "deterministic_requested": _DETERMINISM_STATE.get("deterministic_requested"),
        "deterministic_applied": _DETERMINISM_STATE.get("deterministic_applied"),
        "deterministic_error": _DETERMINISM_STATE.get("deterministic_error"),
        "deterministic_method": _DETERMINISM_STATE.get("deterministic_method"),
        "seed": _DETERMINISM_STATE.get("seed"),
        "pid": os.getpid(),
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: concurrent_harness.py <case_config.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        obs = run_case(cfg)
    except Exception as exc:
        obs = {
            "run_id": cfg.get("run_id"),
            "plan_id": cfg.get("plan_id"),
            "config_id": cfg.get("config_id"),
            "mode": cfg.get("mode"),
            "trials": cfg.get("trials"),
            "execution_status": "SETUP_ERROR",
            "observation_complete": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "runtime": {},
            "baseline": {},
            "tasks": [],
            "task_progress": {"submitted": 0, "started": 0, "completed": 0,
                              "failed": 0, "pending": 0},
            "config": cfg,
        }
    output_path = cfg.get("output_path")
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        tmp = output_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obs, f, ensure_ascii=False, indent=2)
        os.replace(tmp, output_path)
        print(f"OBSERVATION_PATH={output_path}")
    else:
        print("OBSERVATION_JSON=" + json.dumps(obs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
