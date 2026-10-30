"""Concurrent harness runner with capture/baseline/test process separation."""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .case_capture import CaseCapture, count_unsupported_target_cases
from .observation import observation_from_harness, save_observation
from .process_manager import ProcessManager, signal_name
from .schemas import (
    ConcurrentHarnessArtifact,
    ConcurrentTestPlan,
    ExecutionObservation,
    ExecutionStatus,
    write_json,
)

ASAN_SO = "/root/tf_fuzz/lib/python3.10/site-packages/asan_with_fuzzer.so"


def _new_run_id(prefix: str) -> str:
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


class ConcurrentRunner:
    def __init__(
        self,
        output_dir: str | Path,
        python_path: Optional[str] = None,
        preload_path: Optional[str] = ASAN_SO,
        timeout_grace: int = 30,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.python_path = python_path or sys.executable
        self.timeout_grace = timeout_grace
        self.pm = ProcessManager(self.python_path, preload_path=preload_path)
        self.case_capture = CaseCapture(self.output_dir)

    # ── paths ──────────────────────────────────────────────────

    def cases_path(self, plan: ConcurrentTestPlan) -> Path:
        return self.output_dir / "captured_cases" / plan.plan_id / "cases.json"

    def _has_target_calls(self, plan: ConcurrentTestPlan) -> bool:
        p = self.cases_path(plan)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        return any(c.get("target_calls") for c in data.get("cases", []))

    def _run_dirs(self, plan: ConcurrentTestPlan, mode: str, run_id: str) -> Dict[str, Path]:
        d = self.output_dir / "executions" / plan.plan_id / mode / run_id
        d.mkdir(parents=True, exist_ok=True)
        return {
            "dir": d,
            "cfg": d / "case_config.json",
            "obs": d / "observation.json",
            "stdout": d / "stdout.log",
            "stderr": d / "stderr.log",
            "events": d / "events.jsonl",
            "execution": d / "execution.json",
        }

    # ── one mode ───────────────────────────────────────────────

    def _run_mode(
        self,
        artifact: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
        *,
        mode: str,
        runtime: Dict[str, Any],
        run_id: str,
        trials: int,
    ) -> Tuple[ExecutionObservation, Dict[str, Path]]:
        harness = Path(artifact.concurrent_harness).resolve()
        paths = self._run_dirs(plan, mode, run_id)
        config_id = str(runtime.get("config_id") or plan.runtime.get("config_id") or "config")
        cfg = {
            "run_id": run_id,
            "plan_id": plan.plan_id,
            "config_id": config_id,
            "mode": mode,
            "state_model": plan.state_model,
            "trials": int(trials),
            "runtime": runtime,
            "cases_path": str(self.cases_path(plan)),
            "output_path": str(paths["obs"]),
            "events_path": str(paths["events"]),
        }
        write_json(paths["cfg"], cfg)

        pr = self.pm.run(
            [self.python_path, str(harness), str(paths["cfg"])],
            cwd=paths["dir"],
            timeout=plan.timeout_seconds + self.timeout_grace,
        )
        paths["stdout"].write_text(pr.stdout or "", encoding="utf-8")
        paths["stderr"].write_text(pr.stderr or "", encoding="utf-8")

        raw: Optional[Dict[str, Any]] = None
        if paths["obs"].exists():
            try:
                raw = json.loads(paths["obs"].read_text(encoding="utf-8"))
            except Exception:
                raw = None

        if pr.timed_out:
            obs = ExecutionObservation(
                plan_id=plan.plan_id,
                config_id=config_id,
                run_id=run_id,
                mode=mode,
                trials=int(trials),
                execution_status=ExecutionStatus.TIMEOUT,
                process_exit_code=None,
                process_signal="SIGKILL",
                observation_complete=False,
                tasks=[],
                task_progress={"submitted": int(runtime.get("external_threads", 0)),
                               "started": 0, "completed": 0, "failed": 0, "pending": 0},
                runtime={},
                diagnostics={
                    "stdout": str(paths["stdout"]),
                    "stderr": str(paths["stderr"]),
                    "events": str(paths["events"]),
                    "duration": pr.duration,
                },
                harness_path=str(harness),
                config=cfg,
                stdout_path=str(paths["stdout"]),
                stderr_path=str(paths["stderr"]),
            )
        elif raw is not None:
            # Identity check: never accept a stale observation from a previous run.
            if raw.get("run_id") != run_id:
                obs = ExecutionObservation(
                    plan_id=plan.plan_id,
                    config_id=config_id,
                    run_id=run_id,
                    mode=mode,
                    trials=int(trials),
                    execution_status=ExecutionStatus.SETUP_ERROR,
                    process_exit_code=pr.rc,
                    process_signal=pr.signal,
                    observation_complete=False,
                    diagnostics={
                        "stdout": str(paths["stdout"]),
                        "stderr": str(paths["stderr"]),
                        "events": str(paths["events"]),
                        "reason": f"observation run_id mismatch: {raw.get('run_id')} != {run_id}",
                    },
                    harness_path=str(harness),
                    config=cfg,
                    stdout_path=str(paths["stdout"]),
                    stderr_path=str(paths["stderr"]),
                )
            else:
                obs = observation_from_harness(
                    raw,
                    plan_id=plan.plan_id,
                    config_id=config_id,
                    harness_path=str(harness),
                    config=cfg,
                    stdout_path=str(paths["stdout"]),
                    stderr_path=str(paths["stderr"]),
                    process_exit_code=pr.rc,
                    process_signal=pr.signal,
                )
                obs.run_id = run_id
                obs.mode = mode
                obs.trials = int(trials)
                if pr.rc not in (0, None) and obs.execution_status == ExecutionStatus.COMPLETED:
                    obs.execution_status = ExecutionStatus.PARTIAL
        else:
            status = ExecutionStatus.CRASH if (pr.rc is not None and pr.rc < 0) else ExecutionStatus.SETUP_ERROR
            obs = ExecutionObservation(
                plan_id=plan.plan_id,
                config_id=config_id,
                run_id=run_id,
                mode=mode,
                trials=int(trials),
                execution_status=status,
                process_exit_code=pr.rc,
                process_signal=pr.signal or signal_name(pr.rc),
                observation_complete=False,
                tasks=[],
                task_progress={},
                runtime={},
                diagnostics={"stdout": str(paths["stdout"]),
                             "stderr": str(paths["stderr"]),
                             "events": str(paths["events"]),
                             "duration": pr.duration},
                harness_path=str(harness),
                config=cfg,
                stdout_path=str(paths["stdout"]),
                stderr_path=str(paths["stderr"]),
            )
        obs.diagnostics.setdefault("events", str(paths["events"]))
        obs.diagnostics.setdefault("run_dir", str(paths["dir"]))
        save_observation(obs, paths["execution"])
        return obs, paths

    # ── capture ────────────────────────────────────────────────

    def capture_cases(
        self,
        artifact: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not force and self._has_target_calls(plan):
            return {"status": "SKIPPED", "reason": "cases already contain target_calls"}
        run_id = _new_run_id("capture")
        runtime = dict(plan.runtime)
        runtime["external_threads"] = 1
        obs, paths = self._run_mode(
            artifact, plan, mode="capture", runtime=runtime, run_id=run_id, trials=1,
        )
        raw = {}
        if paths["obs"].exists():
            try:
                raw = json.loads(paths["obs"].read_text(encoding="utf-8"))
            except Exception:
                raw = {}
        captured = raw.get("captured_cases") or {}
        merged = self.case_capture.merge_captured_cases(
            plan.plan_id, captured,
            metadata={"capture_run_id": run_id, "capture_status": obs.execution_status.value},
        )
        n_target = sum(1 for c in captured.values() if c.get("target_calls"))
        n_unsupported = count_unsupported_target_cases(captured)
        return {
            "status": "CAPTURED" if n_target else "FALLBACK",
            "unsupported_target_cases": n_unsupported,
            "run_id": run_id,
            "cases_path": str(merged),
            "cases_with_target_calls": n_target,
            "cases_total": len(captured),
            "execution": str(paths["execution"]),
        }

    # ── full baseline + concurrent run ─────────────────────────

    def run(self, artifact: ConcurrentHarnessArtifact, plan: ConcurrentTestPlan) -> ExecutionObservation:
        if not self._has_target_calls(plan):
            self.capture_cases(artifact, plan)

        trials = max(1, int(plan.trials))
        baseline_runtime = dict(plan.runtime)
        baseline_runtime["external_threads"] = 1
        if plan.scenario == "parallel_kernel_execution":
            # Internal-parallel comparison uses one intra/inter-op thread.
            baseline_runtime["intra_op_threads"] = 1
            baseline_runtime["inter_op_threads"] = 1
        baseline_runtime["config_id"] = str(plan.runtime.get("config_id", "config")) + "_baseline"
        baseline_run_id = _new_run_id("baseline")
        baseline_obs, baseline_paths = self._run_mode(
            artifact, plan, mode="baseline", runtime=baseline_runtime,
            run_id=baseline_run_id, trials=trials,
        )

        test_runtime = dict(plan.runtime)
        test_run_id = _new_run_id("test")
        test_obs, test_paths = self._run_mode(
            artifact, plan, mode="concurrent", runtime=test_runtime,
            run_id=test_run_id, trials=trials,
        )

        baseline_ok = (baseline_obs.execution_status == ExecutionStatus.COMPLETED
                       and baseline_obs.observation_complete)
        test_obs.baseline_run_id = baseline_run_id
        test_obs.test_run_id = test_run_id
        test_obs.baseline_runtime = dict(baseline_obs.runtime or {})
        if baseline_ok:
            test_obs.baseline = baseline_obs.baseline or {}
        else:
            test_obs.baseline = {}
            test_obs.observation_complete = False
            test_obs.diagnostics["baseline_failure"] = {
                "run_id": baseline_run_id,
                "execution_status": baseline_obs.execution_status.value,
                "events": str(baseline_paths["events"]),
                "stdout": str(baseline_paths["stdout"]),
                "stderr": str(baseline_paths["stderr"]),
            }
        test_obs.diagnostics["baseline_observation"] = str(baseline_paths["execution"])
        test_obs.diagnostics["test_observation"] = str(test_paths["execution"])
        combined = test_paths["dir"] / "combined_observation.json"
        save_observation(test_obs, combined)
        test_obs.diagnostics["combined_observation"] = str(combined)
        return test_obs

    def run_one_config(self, artifact: ConcurrentHarnessArtifact,
                       plan: ConcurrentTestPlan, config: Dict[str, Any],
                       timeout_seconds: int) -> ExecutionObservation:
        plan = ConcurrentTestPlan.from_dict({**plan.__dict__, "runtime": config,
                                             "timeout_seconds": timeout_seconds})
        return self.run(artifact, plan)
