#!/usr/bin/env python3
"""Generated recipe harness for tf.data / callback concurrency scenarios.

This harness is intentionally independent of the original API harness: the
tf.data and callback scenarios test the *pipeline contract* (cardinality, IDs,
order, progress, callback lifecycle) rather than replaying captured tensor
arguments.  The recipe is supplied in case_config["recipe"].

Supported recipe kinds:
  - tfdata_parallel_map
  - tfdata_interleave
  - tfdata_early_stop
  - python_callback_lifecycle
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _setup_tf(runtime: Dict[str, Any], seed: int = 0):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    intra = int(runtime.get("intra_op_threads", 0) or 0)
    inter = int(runtime.get("inter_op_threads", 0) or 0)
    try:
        if intra > 0:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
        if inter > 0:
            tf.config.threading.set_inter_op_parallelism_threads(inter)
    except Exception:
        pass
    if runtime.get("deterministic", True):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    if seed:
        tf.random.set_seed(int(seed))
    return tf


def _runtime_info(tf, runtime: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "tensorflow_version": getattr(tf, "__version__", None),
        "device": "CPU",
        "gpu_devices": [d.name for d in tf.config.list_physical_devices("GPU")],
        "intra_op_threads": int(runtime.get("intra_op_threads", 0) or 0),
        "inter_op_threads": int(runtime.get("inter_op_threads", 0) or 0),
        "external_threads": int(runtime.get("external_threads", 0) or 0),
        "tf_data_parallel_calls": int(runtime.get("tf_data_parallel_calls", 0) or 0),
        "deterministic_requested": bool(runtime.get("deterministic", True)),
        "seed": None,
    }


def _dataset_record(kind: str, expected_ids: List[int], observed: List[Any],
                    recipe: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    observed_ids = [int(pair[0]) for pair in observed]
    expected_values = {int(i): int(i) * int(i) + 1 for i in expected_ids}
    mismatched = [
        int(i) for i, value in observed
        if expected_values.get(int(i)) != int(value)
    ]
    record = {
        "kind": kind,
        "expected_cardinality": len(expected_ids),
        "observed_cardinality": len(observed),
        "expected_ids": [int(i) for i in expected_ids],
        "observed_ids": observed_ids,
        "missing_ids": sorted(set(expected_ids) - set(observed_ids)),
        "extra_ids": sorted(set(observed_ids) - set(expected_ids)),
        "mismatched_ids": mismatched,
        "deterministic": bool(recipe.get("deterministic", True)),
        "order_preserved": observed_ids == [int(i) for i in expected_ids],
        "parallel_calls": int(recipe.get("parallel_calls", 1) or 1),
        "recipe": _jsonable(recipe),
    }
    if extra:
        record.update(extra)
    return record


def _tfdata_parallel_map(tf, recipe: Dict[str, Any]) -> Dict[str, Any]:
    n = int(recipe.get("num_elements", 64))
    parallel = int(recipe.get("parallel_calls", 4) or 1)
    ids = list(range(n))
    ds = tf.data.Dataset.from_tensor_slices(ids)

    def fn(x):
        x = tf.cast(x, tf.int64)
        return x, tf.math.square(x) + 1

    ds = ds.map(fn, num_parallel_calls=parallel)
    observed = []
    for x, value in ds:
        observed.append((int(x.numpy()), int(value.numpy())))
    return _dataset_record("tfdata_parallel_map", ids, observed, recipe)


def _tfdata_interleave(tf, recipe: Dict[str, Any]) -> Dict[str, Any]:
    sources = int(recipe.get("num_sources", 4))
    per_source = int(recipe.get("per_source", 16))
    parallel = int(recipe.get("parallel_calls", 4) or 1)
    # tf.data.interleave emits round-robin across cycle_length sources; the
    # deterministic order is (0,0), (1,0), ..., (S-1,0), (0,1), ...
    expected = [(source, idx) for idx in range(per_source) for source in range(sources)]

    def make_source(source):
        return tf.data.Dataset.range(per_source).map(
            lambda i: (source, tf.cast(i, tf.int64),
                       tf.math.square(tf.cast(i, tf.int64)) + 1)
        )

    ds = tf.data.Dataset.from_tensor_slices(list(range(sources)))
    ds = ds.interleave(
        make_source,
        cycle_length=sources,
        num_parallel_calls=parallel,
        deterministic=True,
    )
    observed = []
    for source, idx, value in ds:
        observed.append((int(source.numpy()) * 100000 + int(idx.numpy()),
                         int(value.numpy())))
    expected_ids = [s * 100000 + i for s, i in expected]
    expected_values = {s * 100000 + i: i * i + 1 for s, i in expected}
    record = _dataset_record("tfdata_interleave", expected_ids, observed, recipe)
    record["mismatched_ids"] = [
        int(i) for i, value in observed
        if expected_values.get(int(i)) != int(value)
    ]
    record["source_count"] = sources
    record["per_source"] = per_source
    return record


def _tfdata_early_stop(tf, recipe: Dict[str, Any]) -> Dict[str, Any]:
    n = int(recipe.get("num_elements", 64))
    take = min(n, int(recipe.get("take", 8)))
    parallel = int(recipe.get("parallel_calls", 4) or 1)
    ids = list(range(n))

    def build():
        ds = tf.data.Dataset.from_tensor_slices(ids)
        return ds.map(
            lambda x: (tf.cast(x, tf.int64),
                       tf.math.square(tf.cast(x, tf.int64)) + 1),
            num_parallel_calls=parallel,
        )

    taken = []
    for x, value in build().take(take):
        taken.append((int(x.numpy()), int(value.numpy())))
    subsequent_ok = False
    subsequent_count = 0
    try:
        for x, value in build():
            subsequent_count += 1
        subsequent_ok = subsequent_count == n
    except Exception:
        subsequent_ok = False
    # The first phase only promises the taken prefix; do not report the
    # unconsumed tail as missing IDs.
    record = _dataset_record("tfdata_early_stop", ids[:take], taken, recipe)
    record["early_stop"] = {
        "take": take,
        "taken_ids": [i for i, _ in taken],
        "subsequent_pipeline_ok": subsequent_ok,
        "subsequent_count": subsequent_count,
        "expected_subsequent_count": n,
    }
    return record


def _python_callback_lifecycle(tf, recipe: Dict[str, Any]) -> Dict[str, Any]:
    n = int(recipe.get("num_elements", 32))
    parallel = int(recipe.get("parallel_calls", 4) or 1)
    exception_at = recipe.get("exception_at")
    events: List[Dict[str, Any]] = []

    def callback(value):
        element = int(value.numpy())
        events.append({"phase": "enter", "element": element, "ts": time.monotonic()})
        if exception_at is not None and element == int(exception_at):
            events.append({"phase": "exception", "element": element, "ts": time.monotonic()})
            raise ValueError(f"callback exception at element {element}")
        result = int(element) * int(element) + 1
        events.append({"phase": "exit", "element": element, "result": result,
                       "ts": time.monotonic()})
        return tf.constant(result, dtype=tf.int64)

    def build(limit: Optional[int] = None, start: int = 0):
        ids = list(range(start, start + (n if limit is None else limit)))
        ds = tf.data.Dataset.from_tensor_slices(ids)
        return ds.map(
            lambda x: tf.py_function(callback, [x], tf.int64),
            num_parallel_calls=parallel,
        )

    consumed = []
    propagated = False
    error = None
    try:
        for value in build():
            consumed.append(int(value.numpy()))
    except Exception as exc:
        propagated = True
        error = f"{type(exc).__name__}: {exc}"

    subsequent_ok = False
    subsequent_count = 0
    try:
        # Use a fresh range that does not contain the deliberately raising
        # element, otherwise the recovery pipeline would fail for the same
        # expected reason.
        for value in build(limit=min(4, n), start=n + 1):
            subsequent_count += 1
        subsequent_ok = subsequent_count > 0
    except Exception:
        subsequent_ok = False

    entered = [e["element"] for e in events if e.get("phase") == "enter"]
    exited = [e["element"] for e in events if e.get("phase") == "exit"]
    return {
        "kind": "python_callback_lifecycle",
        "callback_count": len(entered),
        "entered_ids": entered,
        "exited_ids": exited,
        "exited_count": len(exited),
        "expected_exception_at": exception_at,
        "exception_propagated": propagated,
        "exception": error,
        "consumed_count": len(consumed),
        "subsequent_pipeline_ok": subsequent_ok,
        "subsequent_count": subsequent_count,
        "events": events[:500],
        "parallel_calls": parallel,
        "recipe": _jsonable(recipe),
    }


def _run_recipe(tf, recipe: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(recipe.get("kind", ""))
    if kind == "tfdata_parallel_map":
        return {"dataset": _tfdata_parallel_map(tf, recipe)}
    if kind == "tfdata_interleave":
        return {"dataset": _tfdata_interleave(tf, recipe)}
    if kind == "tfdata_early_stop":
        return {"dataset": _tfdata_early_stop(tf, recipe)}
    if kind == "python_callback_lifecycle":
        return {"callbacks": _python_callback_lifecycle(tf, recipe)}
    raise ValueError(f"unknown recipe kind: {kind}")


def run_case(case_config: Dict[str, Any]) -> Dict[str, Any]:
    runtime = dict(case_config.get("runtime") or {})
    recipe = dict(case_config.get("recipe") or {})
    seed = int(case_config.get("seed", 0) or 0)
    trials = max(1, int(case_config.get("trials", 1)))
    started = time.time()
    tf = _setup_tf(runtime, seed)
    task: Dict[str, Any] = {
        "task_id": 0,
        "input_id": "recipe",
        "status": "COMPLETED",
        "complete": False,
        "started_at": started,
        "metadata": {"recipe": _jsonable(recipe)},
    }
    try:
        records = []
        for _ in range(trials):
            records.append(_run_recipe(tf, recipe))
        merged: Dict[str, Any] = {"trials": trials}
        datasets = [r["dataset"] for r in records if "dataset" in r]
        callbacks = [r["callbacks"] for r in records if "callbacks" in r]
        if datasets:
            merged["datasets"] = datasets
            merged["dataset"] = datasets[0]
        if callbacks:
            merged["callback_records"] = callbacks
            merged["callbacks"] = callbacks[0]
        task["metadata"].update(merged)
        task["status"] = "COMPLETED"
        task["complete"] = True
    except Exception as exc:
        task["status"] = "EXCEPTION"
        task["complete"] = False
        task["metadata"]["error"] = f"{type(exc).__name__}: {exc}"
        task["metadata"]["traceback"] = traceback.format_exc()
    task["finished_at"] = time.time()
    status = "COMPLETED" if task["status"] == "COMPLETED" else "PARTIAL"
    return {
        "run_id": case_config.get("run_id"),
        "plan_id": case_config.get("plan_id"),
        "config_id": case_config.get("config_id"),
        "mode": "recipe",
        "trials": trials,
        "execution_status": status,
        "observation_complete": task["status"] == "COMPLETED",
        "runtime": _runtime_info(tf, runtime),
        "tasks": [task],
        "task_progress": {
            "submitted": 1, "started": 1,
            "completed": 1 if task["status"] == "COMPLETED" else 0,
            "partial": 0,
            "failed": 0 if task["status"] == "COMPLETED" else 1,
            "pending": 0,
        },
        "diagnostics": {"recipe": _jsonable(recipe)},
        "config": case_config,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: recipe_harness.py <case_config.json>", file=sys.stderr)
        return 2
    case_config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = run_case(case_config)
    out = case_config.get("output_path")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(json.dumps({"execution_status": result.get("execution_status"),
                      "tasks": len(result.get("tasks") or [])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
