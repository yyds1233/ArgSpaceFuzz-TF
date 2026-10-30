#!/usr/bin/env python3
"""Generated concurrent harness template for one TensorFlow API.

This file is copied by concurrent_oracle.transformer and has three placeholders
replaced with JSON literals:
  __ORIGINAL_HARNESS__  -> original fuzz_api_one harness path
  __TARGET_PATCHES__    -> list of dotted target call names, e.g. tf.math.log1p
  __DEFAULT_CASES__     -> list of base64-encoded input byte strings
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
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


# ── serialization ──────────────────────────────────────────────

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
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"__bytes_b64__": base64.b64encode(value).decode("ascii")}
    return value


def _describe(value: Any) -> Dict[str, Any]:
    """Return a JSON-safe description of a target call output."""
    import hashlib
    import numpy as np
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        arr = value.numpy()
        desc = {
            "kind": "tensor",
            "dtype": str(value.dtype),
            "shape": [int(x) for x in value.shape],
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        }
        if int(arr.size) <= 4096:
            desc["value"] = _jsonable(arr)
        else:
            desc["value_truncated"] = True
        return desc

    if isinstance(value, tf.SparseTensor):
        return {
            "kind": "sparse_tensor",
            "dtype": str(value.dtype),
            "dense_shape": [int(x) for x in value.dense_shape.numpy().tolist()],
            "indices": _jsonable(value.indices.numpy()),
            "values": _jsonable(value.values.numpy()),
        }

    if isinstance(value, tf.RaggedTensor):
        return {
            "kind": "ragged_tensor",
            "dtype": str(value.dtype),
            "flat_values": _describe(value.flat_values),
            "row_splits": _jsonable(value.row_splits.numpy()),
        }

    if isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "value": _jsonable(value) if value.size <= 4096 else None,
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


# ── monkey patching ────────────────────────────────────────────

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
                try:
                    out = orig(*args, **kwargs)
                except Exception as exc:
                    if rec is not None:
                        rec["exceptions"].append({
                            "call": call_name,
                            "type": type(exc).__name__,
                            "message": str(exc),
                        })
                    raise
                if rec is not None:
                    rec["outputs"].append({
                        "call": call_name,
                        "output": _safe_describe(out),
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

def _setup_tf(runtime: Dict[str, Any]):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    intra = int(runtime.get("intra_op_threads", 1))
    inter = int(runtime.get("inter_op_threads", 1))
    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
    except Exception:
        # The setting must happen before runtime initialization; if it fails,
        # report it in the observation rather than silently continuing.
        raise
    gpus = tf.config.list_physical_devices("GPU")
    if runtime.get("require_cpu_only", True) and gpus:
        raise RuntimeError(f"CPU-only test expected, but GPU devices are visible: {gpus}")
    return tf


def _load_cases(case_config: Dict[str, Any]) -> List[Dict[str, str]]:
    path = case_config.get("cases_path")
    if path and os.path.exists(path):
        data = json.loads(open(path, "r", encoding="utf-8").read())
        cases = data.get("cases", [])
        out = []
        for i, c in enumerate(cases):
            if "data_b64" in c:
                out.append({"case_id": c.get("case_id", f"case_{i:03d}"), "data_b64": c["data_b64"]})
            elif "data" in c:
                out.append({"case_id": c.get("case_id", f"case_{i:03d}"),
                            "data_b64": base64.b64encode(bytes(c["data"])).decode("ascii")})
        if out:
            return out
    return [{"case_id": f"case_{i:03d}", "data_b64": x} for i, x in enumerate(DEFAULT_CASES)]


def _decode(data_b64: str) -> bytes:
    return base64.b64decode(data_b64.encode("ascii"))


# ── execution ──────────────────────────────────────────────────

def _invoke(module, case: Dict[str, str], task_id: int) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "task_id": task_id,
        "input_id": case["case_id"],
        "status": "RUNNING",
        "outputs": [],
        "exceptions": [],
        "started_at": time.time(),
        "finished_at": None,
    }
    _PATCH_STATE.record = rec
    try:
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
            "call": None,
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        rec["finished_at"] = time.time()
        _PATCH_STATE.record = None
    return rec


def _worker(module, task_id: int, case: Dict[str, str], barrier, results, lock):
    rec = {
        "task_id": task_id,
        "input_id": case["case_id"],
        "status": "PENDING",
        "outputs": [],
        "exceptions": [],
        "started_at": None,
        "finished_at": None,
    }
    try:
        barrier.wait(timeout=30)
        rec["started_at"] = time.time()
        item = _invoke(module, case, task_id)
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
        with lock:
            results.append(rec)


def run_case(case_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(case_config.get("runtime") or {})
    tf = _setup_tf(runtime)

    spec = importlib.util.spec_from_file_location(
        "concurrent_original_harness", ORIGINAL_HARNESS
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load original harness: {ORIGINAL_HARNESS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    originals = _patch_targets(tf)
    try:
        cases = _load_cases(case_config)
        baseline: Dict[str, Any] = {}
        for i, case in enumerate(cases):
            baseline[case["case_id"]] = _invoke(module, case, i)

        if case_config.get("capture_only"):
            return {
                "plan_id": case_config.get("plan_id"),
                "config_id": case_config.get("config_id"),
                "execution_status": "COMPLETED",
                "observation_complete": True,
                "runtime": _runtime_info(tf, runtime),
                "baseline": baseline,
                "tasks": [],
                "task_progress": {"submitted": 0, "started": 0, "completed": 0,
                                  "failed": 0, "pending": 0},
                "config": case_config,
            }

        n = max(1, int(runtime.get("external_threads", 2)))
        selected = []
        for i in range(n):
            selected.append(cases[i % len(cases)])

        barrier = threading.Barrier(n)
        results: List[Dict[str, Any]] = []
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=n) as executor:
            futures = [
                executor.submit(_worker, module, task_id, case, barrier, results, lock)
                for task_id, case in enumerate(selected)
            ]
            for f in futures:
                f.result()

        results.sort(key=lambda x: x["task_id"])
        completed = sum(1 for r in results if r.get("status") == "COMPLETED")
        failed = sum(1 for r in results if r.get("status") in ("EXCEPTION", "BARRIER_ERROR"))
        progress = {
            "submitted": n,
            "started": sum(1 for r in results if r.get("started_at") is not None),
            "completed": completed,
            "failed": failed,
            "pending": max(0, n - len(results)),
        }
        return {
            "plan_id": case_config.get("plan_id"),
            "config_id": case_config.get("config_id"),
            "execution_status": "COMPLETED",
            "observation_complete": len(results) == n,
            "runtime": _runtime_info(tf, runtime),
            "baseline": baseline,
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
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: concurrent_harness.py <case_config.json>", file=sys.stderr)
        return 2
    cfg = json.loads(open(sys.argv[1], "r", encoding="utf-8").read())
    try:
        obs = run_case(cfg)
    except Exception as exc:
        obs = {
            "plan_id": cfg.get("plan_id"),
            "config_id": cfg.get("config_id"),
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
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(obs, f, ensure_ascii=False, indent=2)
        print(f"OBSERVATION_PATH={output_path}")
    else:
        print("OBSERVATION_JSON=" + json.dumps(obs, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
