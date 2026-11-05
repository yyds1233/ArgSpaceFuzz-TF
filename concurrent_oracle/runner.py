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


def baseline_covers_tasks(
    baseline_obs: ExecutionObservation,
    test_obs: ExecutionObservation,
) -> bool:
    """A baseline is valid only for the cases the test actually uses."""
    if baseline_obs.execution_status != ExecutionStatus.COMPLETED:
        return False
    used_input_ids = {t.input_id for t in (test_obs.tasks or [])}
    baseline = baseline_obs.baseline or {}
    return all((baseline.get(input_id) or {}).get("complete") for input_id in used_input_ids)


def _baseline_bad_cases(obs: ExecutionObservation) -> list[str]:
    return [
        case_id
        for case_id, entry in (obs.baseline or {}).items()
        if not isinstance(entry, dict) or not entry.get("complete")
    ]


class ConcurrentRunner:
    def __init__(
        self,
        output_dir: str | Path,
        python_path: Optional[str] = None,
        preload_path: Optional[str] = ASAN_SO,
        timeout_grace: int = 30,
        no_progress_timeout: float = 20.0,
        process_repetitions: int = 1,
        sanitizer: Optional[Dict[str, Any]] = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.python_path = python_path or sys.executable
        self.timeout_grace = timeout_grace
        self.no_progress_timeout = no_progress_timeout
        self.process_repetitions = max(1, int(process_repetitions))
        self.sanitizer = dict(sanitizer or {})
        effective_preload = preload_path
        if self.sanitizer:
            if not self.sanitizer.get("enabled"):
                effective_preload = None
            elif self.sanitizer.get("preload_path"):
                effective_preload = self.sanitizer["preload_path"]
        self.pm = ProcessManager(self.python_path, preload_path=effective_preload)
        self.case_capture = CaseCapture(self.output_dir)

    # ── paths ──────────────────────────────────────────────────

    def cases_path(self, plan: ConcurrentTestPlan) -> Path:
        return self.output_dir / "captured_cases" / plan.plan_id / "cases.json"

    def _has_eligible_cases(self, plan: ConcurrentTestPlan) -> bool:
        p = self.cases_path(plan)
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        cases = data.get("cases", [])
        flagged = [c for c in cases if "eligible" in c]
        if flagged:
            return any(c.get("eligible") for c in flagged)
        return any(c.get("target_calls") for c in cases)

    def _sanitizer_report(self, stderr: str, exit_code: Optional[int],
                          signal: Optional[str], stderr_path: Optional[Path]) -> Dict[str, Any]:
        """Parse ASan/TSan/UBSan/MSan output into an O9 evidence record."""
        enabled = bool(self.sanitizer.get("enabled"))
        stype = str(self.sanitizer.get("type", "none") or "none")
        text = stderr or ""
        explicit_instrumented = self.sanitizer.get("instrumented")
        if explicit_instrumented is None:
            # A preload library alone does not prove the target was compiled
            # with sanitizer instrumentation; require a build id for PASS.
            instrumented = bool(self.sanitizer.get("build_id"))
        else:
            instrumented = bool(explicit_instrumented)
        report = {
            "enabled": enabled,
            "type": stype,
            "detected": False,
            "instrumented": instrumented,
            "build_id": self.sanitizer.get("build_id"),
            "raw_path": str(stderr_path) if stderr_path else None,
            "exit_code": exit_code,
            "signal": signal,
            "summary": None,
            "stack": [],
        }
        if not enabled:
            return report
        patterns = [
            ("AddressSanitizer", "ERROR: AddressSanitizer:", "AddressSanitizer"),
            ("ThreadSanitizer", "WARNING: ThreadSanitizer:", "ThreadSanitizer"),
            ("MemorySanitizer", "WARNING: MemorySanitizer:", "MemorySanitizer"),
            ("UndefinedBehaviorSanitizer", "runtime error:", "UBSan"),
        ]
        for report_type, needle, short in patterns:
            if needle in text:
                report.update({"detected": True, "report_type": short})
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if needle in line:
                        report["summary"] = line.strip()[:1000]
                        report["stack"] = [x.strip()[:500] for x in lines[i: i + 20]
                                           if x.strip().startswith("#")]
                        break
                break
        return report

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
        case_offset: int = 0,
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
            "seed": int(plan.metadata.get("seed", 0)),
            "trials": int(trials),
            "runtime": runtime,
            "cases_path": str(self.cases_path(plan)),
            "case_offset": int(case_offset),
            "backend": (plan.metadata or {}).get("backend"),
            "invalid_arg": (plan.metadata or {}).get("invalid_arg"),
            "recipe": (plan.metadata or {}).get("recipe"),
            "output_path": str(paths["obs"]),
            "events_path": str(paths["events"]),
        }
        write_json(paths["cfg"], cfg)

        env_extra: Dict[str, str] = {}
        if self.sanitizer.get("enabled") and self.sanitizer.get("options"):
            stype = str(self.sanitizer.get("type", "")).lower()
            if "tsan" in stype:
                env_extra["TSAN_OPTIONS"] = str(self.sanitizer["options"])
            elif "msan" in stype:
                env_extra["MSAN_OPTIONS"] = str(self.sanitizer["options"])
            else:
                env_extra["ASAN_OPTIONS"] = str(self.sanitizer["options"])
        pr = self.pm.run(
            [self.python_path, str(harness), str(paths["cfg"])],
            cwd=paths["dir"],
            timeout=plan.timeout_seconds + self.timeout_grace,
            events_path=paths["events"],
            no_progress_timeout=self.no_progress_timeout,
            env_extra=env_extra or None,
        )
        paths["stdout"].write_text(pr.stdout or "", encoding="utf-8")
        paths["stderr"].write_text(pr.stderr or "", encoding="utf-8")
        sanitizer_text = pr.stderr or ""
        sanitizer_path = paths["stderr"]
        log_path = self.sanitizer.get("log_path")
        if self.sanitizer.get("enabled") and log_path and Path(log_path).exists():
            try:
                sanitizer_text += "\n" + Path(log_path).read_text(
                    encoding="utf-8", errors="replace"
                )
                sanitizer_path = Path(log_path)
            except Exception:
                pass
        sanitizer_report = self._sanitizer_report(
            sanitizer_text, pr.rc, pr.signal, sanitizer_path
        )

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
                    "no_progress": pr.no_progress,
                    "last_event": pr.last_event,
                    "last_event_age": pr.last_event_age,
                    "stack_signal_sent": pr.stack_signal_sent,
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
                             "duration": pr.duration,
                             "no_progress": pr.no_progress,
                             "last_event": pr.last_event,
                             "last_event_age": pr.last_event_age,
                             "stack_signal_sent": pr.stack_signal_sent},
                harness_path=str(harness),
                config=cfg,
                stdout_path=str(paths["stdout"]),
                stderr_path=str(paths["stderr"]),
            )
        obs.diagnostics.setdefault("events", str(paths["events"]))
        obs.diagnostics.setdefault("run_dir", str(paths["dir"]))
        obs.diagnostics["sanitizer"] = sanitizer_report
        save_observation(obs, paths["execution"])
        return obs, paths

    # ── capture ────────────────────────────────────────────────

    def capture_cases(
        self,
        artifact: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
        force: bool = False,
    ) -> Dict[str, Any]:
        if not force and self._has_eligible_cases(plan):
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
        merged_data = json.loads(merged.read_text(encoding="utf-8"))
        merged_cases = merged_data.get("cases", [])
        n_target = sum(1 for c in merged_cases if c.get("target_calls"))
        eligible = [c for c in merged_cases if c.get("eligible")]
        ineligible = [c for c in merged_cases if "eligible" in c and not c.get("eligible")]
        n_unsupported = count_unsupported_target_cases(captured)
        conversion_cases = [
            {"case_id": c.get("case_id"), "conversions": c.get("arg_conversions")}
            for c in merged_cases if c.get("arg_conversions")
        ]
        return {
            "status": "CAPTURED" if eligible else "FALLBACK",
            "unsupported_target_cases": n_unsupported,
            "run_id": run_id,
            "cases_path": str(merged),
            "cases_with_target_calls": n_target,
            "cases_eligible": len(eligible),
            "cases_ineligible": len(ineligible),
            "ineligible_reasons": [
                {"case_id": c.get("case_id"), "reason": c.get("ineligible_reason")}
                for c in ineligible[:20]
            ],
            "conversion_cases": conversion_cases,
            "cases_total": len(merged_cases),
            "execution": str(paths["execution"]),
        }

    # ── full baseline + concurrent run ─────────────────────────

    def run(self, artifact: ConcurrentHarnessArtifact, plan: ConcurrentTestPlan) -> ExecutionObservation:
        backend = str((plan.metadata or {}).get("backend", ""))
        if backend in {"tfdata_recipe", "callback_recipe"}:
            run_id = _new_run_id("recipe")
            obs, paths = self._run_mode(
                artifact, plan, mode="recipe", runtime=dict(plan.runtime),
                run_id=run_id, trials=max(1, int(plan.trials)), case_offset=0,
            )
            obs.test_run_id = run_id
            obs.mode = "recipe"
            obs.diagnostics["backend"] = backend
            obs.diagnostics["test_observation"] = str(paths["execution"])
            combined = paths["dir"] / "combined_observation.json"
            save_observation(obs, combined)
            obs.diagnostics["combined_observation"] = str(combined)
            return obs

        if backend in {"invalid_input_replay", "exception_recovery_replay"}:
            if not self._has_eligible_cases(plan):
                self.capture_cases(artifact, plan)
            mode = "invalid_input" if backend == "invalid_input_replay" else "exception_recovery"
            run_id = _new_run_id(mode)
            obs, paths = self._run_mode(
                artifact, plan, mode=mode, runtime=dict(plan.runtime),
                run_id=run_id, trials=1, case_offset=0,
            )
            obs.test_run_id = run_id
            obs.mode = mode
            obs.diagnostics["backend"] = backend
            obs.diagnostics["test_observation"] = str(paths["execution"])
            combined = paths["dir"] / "combined_observation.json"
            save_observation(obs, combined)
            obs.diagnostics["combined_observation"] = str(combined)
            return obs

        if not self._has_eligible_cases(plan):
            self.capture_cases(artifact, plan)

        trials = max(1, int(plan.trials))
        baseline_runtime = dict(plan.runtime)
        baseline_runtime["external_threads"] = 1
        if plan.scenario == "parallel_kernel_execution":
            # Internal-parallel comparison uses one intra/inter-op thread.
            baseline_runtime["intra_op_threads"] = 1
            baseline_runtime["inter_op_threads"] = 1
        baseline_runtime["config_id"] = str(plan.runtime.get("config_id", "config")) + "_baseline"

        baseline_obs: Optional[ExecutionObservation] = None
        baseline_paths: Optional[Dict[str, Path]] = None
        baseline_run_id: Optional[str] = None
        baseline_excluded: list[str] = []
        baseline_attempts = 0
        for _attempt in range(3):
            baseline_attempts += 1
            baseline_run_id = _new_run_id("baseline")
            baseline_obs, baseline_paths = self._run_mode(
                artifact, plan, mode="baseline", runtime=baseline_runtime,
                run_id=baseline_run_id, trials=trials,
            )
            bad = _baseline_bad_cases(baseline_obs)
            if not bad:
                break
            changed = self.case_capture.mark_ineligible(
                plan.plan_id, bad, "baseline_not_complete"
            )
            baseline_excluded.extend(bad)
            if not changed:
                break
        assert baseline_obs is not None and baseline_paths is not None

        test_runtime = dict(plan.runtime)
        reps = max(1, int(self.process_repetitions))
        n_threads = max(1, int(test_runtime.get("external_threads", 1)))
        test_runs = []
        for rep in range(reps):
            rep_run_id = _new_run_id(f"test_rep{rep}")
            rep_obs, rep_paths = self._run_mode(
                artifact, plan, mode="concurrent", runtime=test_runtime,
                run_id=rep_run_id, trials=trials, case_offset=rep * n_threads,
            )
            test_runs.append((rep, rep_run_id, rep_obs, rep_paths))

        if reps == 1:
            _rep, test_run_id, test_obs, test_paths = test_runs[0]
            test_obs.diagnostics["process_repetitions"] = 1
            test_obs.diagnostics["test_run_ids"] = [test_run_id]
        else:
            test_obs, test_paths = self._merge_test_runs(plan, test_runs, trials)
            test_run_id = test_obs.test_run_id

        baseline_ok = baseline_covers_tasks(baseline_obs, test_obs)
        test_obs.baseline_run_id = baseline_run_id
        test_obs.test_run_id = test_run_id
        test_obs.baseline_runtime = dict(baseline_obs.runtime or {})
        test_obs.diagnostics["process_repetitions"] = reps
        test_obs.diagnostics["baseline_qualification"] = {
            "attempts": baseline_attempts,
            "excluded_cases": sorted(set(baseline_excluded)),
        }
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

    def _merge_test_runs(
        self,
        plan: ConcurrentTestPlan,
        test_runs,
        trials: int,
    ) -> Tuple[ExecutionObservation, Dict[str, Path]]:
        """Merge repeated concurrent processes into one observation."""
        _rep0, _run0, base_obs, base_paths = test_runs[0]
        tasks = []
        statuses = []
        test_run_ids = []
        progress = {"submitted": 0, "started": 0, "completed": 0,
                    "partial": 0, "failed": 0, "pending": 0}
        for _rep, run_id, obs, _paths in test_runs:
            statuses.append(obs.execution_status.value)
            test_run_ids.append(run_id)
            for key in progress:
                progress[key] += int((obs.task_progress or {}).get(key, 0) or 0)
            for task in obs.tasks or []:
                task.task_id = len(tasks)
                tasks.append(task)
        complete = all(bool(obs.observation_complete) for _r, _id, obs, _p in test_runs)
        if all(s == ExecutionStatus.COMPLETED.value for s in statuses):
            execution_status = ExecutionStatus.COMPLETED
        else:
            execution_status = next(
                (ExecutionStatus(s) for s in statuses if s != ExecutionStatus.COMPLETED.value),
                ExecutionStatus.PARTIAL,
            )
        base_obs.run_id = base_obs.run_id or test_run_ids[0]
        base_obs.test_run_id = test_run_ids[0]
        base_obs.execution_status = execution_status
        base_obs.observation_complete = complete
        base_obs.tasks = tasks
        base_obs.task_progress = progress
        base_obs.diagnostics["process_repetitions"] = len(test_runs)
        base_obs.diagnostics["test_run_ids"] = test_run_ids
        base_obs.diagnostics["test_statuses"] = statuses
        base_obs.diagnostics["test_events"] = [
            str(p["events"]) for _r, _id, _o, p in test_runs
        ]
        return base_obs, base_paths

    def run_one_config(self, artifact: ConcurrentHarnessArtifact,
                       plan: ConcurrentTestPlan, config: Dict[str, Any],
                       timeout_seconds: int) -> ExecutionObservation:
        plan = ConcurrentTestPlan.from_dict({**plan.__dict__, "runtime": config,
                                             "timeout_seconds": timeout_seconds})
        return self.run(artifact, plan)
