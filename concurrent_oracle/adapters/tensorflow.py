"""TensorFlow-specific adapter: CPU checks, materialization, comparison."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import FrameworkAdapter


def _as_list(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:
        np = None
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return type(value)(_as_list(v) for v in value)
    if isinstance(value, dict):
        return {k: _as_list(v) for k, v in value.items()}
    return value


class TensorFlowAdapter(FrameworkAdapter):
    def __init__(self, comparison: Optional[Dict[str, Any]] = None):
        self.comparison = comparison or {
            "float_atol": 1e-6,
            "float_rtol": 1e-5,
            "exact_integer": True,
            "compare_nan_mask": True,
            "compare_inf_sign": True,
        }
        self._tf = None

    def _tensorflow(self):
        if self._tf is None:
            import tensorflow as tf
            self._tf = tf
        return self._tf

    def validate_environment(self) -> Dict[str, Any]:
        tf = self._tensorflow()
        gpus = tf.config.list_physical_devices("GPU")
        return {
            "tensorflow_version": getattr(tf, "__version__", None),
            "tensorflow_path": getattr(tf, "__file__", None),
            "gpu_devices": [str(x) for x in gpus],
            "cpu_only": len(gpus) == 0,
        }

    def configure_runtime(self, config: Dict[str, Any]) -> Dict[str, Any]:
        # In the parent this is informational; the generated harness performs
        # the real CUDA_VISIBLE_DEVICES/import/thread-config ordering.
        return {
            "device": config.get("device", "CPU"),
            "intra_op_threads": int(config.get("intra_op_threads", 1)),
            "inter_op_threads": int(config.get("inter_op_threads", 1)),
            "external_threads": int(config.get("external_threads", 1)),
        }

    def load_case_inputs(self, case_path: str):
        p = Path(case_path)
        if p.is_dir():
            p = p / "cases.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for c in data.get("cases", []):
            if "data_b64" in c:
                out.append({"case_id": c.get("case_id"), "data": base64.b64decode(c["data_b64"])})
            elif "data" in c:
                out.append({"case_id": c.get("case_id"), "data": bytes(c["data"])})
        return out

    def materialize_output(self, output):
        tf = self._tensorflow()
        import numpy as np
        if isinstance(output, tf.Tensor):
            return output.numpy()
        if isinstance(output, tf.SparseTensor):
            return {
                "indices": output.indices.numpy(),
                "values": output.values.numpy(),
                "dense_shape": output.dense_shape.numpy(),
            }
        if isinstance(output, tf.RaggedTensor):
            return {
                "flat_values": output.flat_values.numpy(),
                "row_splits": output.row_splits.numpy(),
            }
        if isinstance(output, np.ndarray):
            return output
        if isinstance(output, (list, tuple)):
            return type(output)(self.materialize_output(x) for x in output)
        if isinstance(output, dict):
            return {k: self.materialize_output(v) for k, v in output.items()}
        return output

    def compare_outputs(self, expected, actual, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = dict(self.comparison)
        if config:
            cfg.update(config)
        diffs: List[str] = []
        self._compare(expected, actual, "$", cfg, diffs)
        return {"equal": not diffs, "diffs": diffs[:50], "reason": "ok" if not diffs else "; ".join(diffs[:5])}

    def _compare(self, a, b, path: str, cfg: Dict[str, Any], diffs: List[str]) -> None:
        try:
            import numpy as np
        except Exception:
            np = None
        if np is not None:
            if isinstance(a, np.generic):
                a = a.item()
            if isinstance(b, np.generic):
                b = b.item()
            if isinstance(a, np.ndarray):
                a = a.tolist()
            if isinstance(b, np.ndarray):
                b = b.tolist()
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a.keys()) != set(b.keys()):
                diffs.append(f"{path}: dict keys differ {sorted(a)} vs {sorted(b)}")
                return
            for k in a:
                self._compare(a[k], b[k], f"{path}.{k}", cfg, diffs)
            return
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b):
                diffs.append(f"{path}: length {len(a)} != {len(b)}")
                return
            for i, (x, y) in enumerate(zip(a, b)):
                self._compare(x, y, f"{path}[{i}]", cfg, diffs)
            return
        if isinstance(a, float) or isinstance(b, float):
            af, bf = float(a), float(b)
            nan_a, nan_b = math.isnan(af), math.isnan(bf)
            if nan_a or nan_b:
                if cfg.get("compare_nan_mask", True) and nan_a != nan_b:
                    diffs.append(f"{path}: NaN mask differs ({af} vs {bf})")
                return
            inf_a = math.isinf(af)
            inf_b = math.isinf(bf)
            if inf_a or inf_b:
                if cfg.get("compare_inf_sign", True) and (af != bf):
                    diffs.append(f"{path}: inf sign/value differs ({af} vs {bf})")
                return
            atol = float(cfg.get("float_atol", 1e-6))
            rtol = float(cfg.get("float_rtol", 1e-5))
            if abs(af - bf) > atol + rtol * abs(bf):
                diffs.append(f"{path}: float differs ({af} vs {bf}, atol={atol}, rtol={rtol})")
            return
        if isinstance(a, int) and isinstance(b, int):
            if cfg.get("exact_integer", True) and a != b:
                diffs.append(f"{path}: int differs ({a} vs {b})")
            return
        if a != b:
            diffs.append(f"{path}: value differs ({a!r} vs {b!r})")

    def get_runtime_info(self) -> Dict[str, Any]:
        tf = self._tensorflow()
        return {
            "tensorflow_version": getattr(tf, "__version__", None),
            "gpu_devices": [str(x) for x in tf.config.list_physical_devices("GPU")],
        }

    def cleanup(self) -> None:
        return None
