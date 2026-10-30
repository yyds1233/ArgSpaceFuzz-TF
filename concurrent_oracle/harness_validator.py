"""Validation of generated concurrent harnesses and serial baselines."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .schemas import (
    ConcurrentHarnessArtifact,
    ConcurrentTestPlan,
    ValidationStatus,
)

ASAN_SO = "/root/tf_fuzz/lib/python3.10/site-packages/asan_with_fuzzer.so"


class HarnessValidator:
    def __init__(self, output_dir: Optional[str | Path] = None, python_path: Optional[str] = None):
        self.output_dir = Path(output_dir).resolve() if output_dir else None
        self.python_path = python_path or sys.executable

    def validate_syntax(self, path: str | Path) -> Dict[str, Any]:
        p = Path(path)
        try:
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            return {"ok": True, "reason": "ok"}
        except SyntaxError as exc:
            return {"ok": False, "reason": f"SyntaxError: {exc}"}
        except Exception as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

    def validate_imports(self, path: str | Path, timeout: int = 30) -> Dict[str, Any]:
        p = Path(path).resolve()
        code = (
            "import importlib.util, sys\n"
            f"p = {str(p)!r}\n"
            "spec = importlib.util.spec_from_file_location('co_validate', p)\n"
            "assert spec and spec.loader\n"
            "m = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(m)\n"
            "print('IMPORT_OK')\n"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        try:
            r = subprocess.run([self.python_path, "-c", code], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, timeout=timeout, env=env)
            ok = r.returncode == 0 and "IMPORT_OK" in (r.stdout or "")
            return {"ok": ok, "rc": r.returncode,
                    "stdout": (r.stdout or "")[-2000:], "stderr": (r.stderr or "")[-2000:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "rc": None, "reason": "import timeout"}

    def validate_target_call(self, path: str | Path, api_name: str) -> Dict[str, Any]:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            return {"ok": False, "reason": f"SyntaxError: {exc}"}
        patches: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "TARGET_PATCHES":
                        try:
                            value = ast.literal_eval(node.value)
                            if isinstance(value, list):
                                patches = [str(x) for x in value]
                        except Exception:
                            pass
        ok = bool(patches) and any(
            api_name == x or api_name.split(".")[-1].lower() == x.split(".")[-1].lower()
            for x in patches
        )
        return {"ok": ok, "patches": patches, "reason": "ok" if ok else "target API marker not found in TARGET_PATCHES"}

    def validate_worker_structure(self, path: str | Path) -> Dict[str, Any]:
        text = Path(path).read_text(encoding="utf-8")
        checks = {
            "thread_pool": "ThreadPoolExecutor" in text,
            "barrier": "Barrier" in text,
            "task_id": "task_id" in text,
            "exception_capture": "except Exception" in text,
            "worker_finish": "finished_at" in text,
            "task_progress": "task_progress" in text,
        }
        ok = all(checks.values())
        return {"ok": ok, "checks": checks,
                "reason": "ok" if ok else "missing: " + ", ".join(k for k, v in checks.items() if not v)}

    def run_serial_baseline(
        self,
        harness_path: str | Path,
        plan: ConcurrentTestPlan,
        output_dir: str | Path,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        harness = Path(harness_path).resolve()
        out_dir = Path(output_dir).resolve()
        cases_path = out_dir / "captured_cases" / plan.plan_id / "cases.json"
        run_id = f"serial_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        serial_dir = out_dir / "executions" / plan.plan_id / "serial_baseline" / run_id
        serial_dir.mkdir(parents=True, exist_ok=True)
        obs_path = serial_dir / "observation.json"
        cfg_path = serial_dir / "case_config.json"
        events_path = serial_dir / "events.jsonl"
        runtime = dict(plan.runtime)
        runtime.update({
            "external_threads": 1,
            "device": "CPU",
            "require_cpu_only": True,
        })
        if plan.scenario == "parallel_kernel_execution":
            runtime["intra_op_threads"] = 1
            runtime["inter_op_threads"] = 1
        cfg = {
            "run_id": run_id,
            "plan_id": plan.plan_id,
            "config_id": "serial_baseline",
            "mode": "baseline",
            "state_model": plan.state_model,
            "seed": int(plan.metadata.get("seed", 0)),
            "trials": 1,
            "runtime": runtime,
            "cases_path": str(cases_path),
            "output_path": str(obs_path),
            "events_path": str(events_path),
        }
        tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
        tmp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cfg_path)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        if Path(ASAN_SO).exists():
            env["LD_PRELOAD"] = ASAN_SO
            env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
        try:
            r = subprocess.run(
                [self.python_path, str(harness), str(cfg_path)],
                cwd=str(serial_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout or (plan.timeout_seconds + 30),
                env=env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired as exc:
            return {"ok": False, "rc": None, "reason": "serial baseline timeout",
                    "run_id": run_id,
                    "stdout": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
                    "stderr": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
                    "events": str(events_path)}
        obs = None
        if obs_path.exists():
            try:
                obs = json.loads(obs_path.read_text(encoding="utf-8"))
            except Exception:
                obs = None
        if obs is None:
            return {"ok": False, "rc": r.returncode, "reason": "observation file missing",
                    "run_id": run_id,
                    "stdout": (r.stdout or "")[-2000:], "stderr": (r.stderr or "")[-2000:],
                    "events": str(events_path)}
        if obs.get("run_id") != run_id:
            return {"ok": False, "rc": r.returncode, "reason": "observation run_id mismatch",
                    "run_id": run_id, "observation_path": str(obs_path),
                    "events": str(events_path)}
        baseline = obs.get("baseline") or {}
        completed = [k for k, v in baseline.items()
                     if isinstance(v, dict) and v.get("status") == "COMPLETED" and v.get("outputs")]
        ok = (obs.get("execution_status") == "COMPLETED"
              and bool(obs.get("observation_complete"))
              and bool(completed))
        return {
            "ok": ok,
            "rc": r.returncode,
            "run_id": run_id,
            "reason": "ok" if ok else f"execution_status={obs.get('execution_status')} completed_cases={len(completed)}",
            "completed_cases": completed,
            "observation_path": str(obs_path),
            "events": str(events_path),
            "observation": obs,
            "stdout": (r.stdout or "")[-2000:],
            "stderr": (r.stderr or "")[-2000:],
        }

    def validate(
        self,
        artifact: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
        output_dir: Optional[str | Path] = None,
        run_baseline: bool = True,
    ) -> Dict[str, Any]:
        path = Path(artifact.concurrent_harness)
        result: Dict[str, Any] = {
            "plan_id": plan.plan_id,
            "syntax": self.validate_syntax(path),
            "imports": self.validate_imports(path),
            "target_call": self.validate_target_call(path, plan.metadata.get("api_name", "")),
            "worker_structure": self.validate_worker_structure(path),
        }
        ok = all(result[k]["ok"] for k in ("syntax", "imports", "target_call", "worker_structure"))
        if ok and run_baseline:
            out = Path(output_dir).resolve() if output_dir else self.output_dir
            if out is None:
                out = path.parent.parent.parent
            result["serial_baseline"] = self.run_serial_baseline(path, plan, out)
            ok = ok and result["serial_baseline"]["ok"]
        result["ok"] = ok
        artifact.validation_status = ValidationStatus.PASSED if ok else ValidationStatus.FAILED
        artifact.validation_record = result
        return result
