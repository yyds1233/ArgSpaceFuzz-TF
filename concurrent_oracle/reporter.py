"""JSON and Markdown reporting for OracleResult."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from .schemas import (
    ConcurrentTestPlan,
    ExecutionObservation,
    HarnessArtifact,
    OracleResult,
    OracleStatus,
    to_jsonable,
    write_json,
)


class Reporter:
    def write_json(self, result: OracleResult, path: str | Path) -> Path:
        return write_json(path, result)

    def write_markdown(
        self,
        result: OracleResult,
        path: str | Path,
        artifact: Optional[HarnessArtifact] = None,
        plan: Optional[ConcurrentTestPlan] = None,
        observation: Optional[ExecutionObservation] = None,
    ) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Concurrent Oracle Report",
            "",
            "## 1. Basic Information",
            "",
            f"- plan_id: `{result.plan_id}`",
            f"- scenario: `{result.scenario}`",
            f"- execution_status: **{result.execution_status.value}**",
        ]
        if artifact:
            lines += [
                f"- api: `{artifact.api_name}`",
                f"- original harness: `{artifact.harness_path}`",
                f"- harness hash: `{artifact.harness_hash}`",
                f"- psr: `{artifact.psr_path}`",
            ]
        lines += ["", "## 2. Test Plan", ""]
        if plan:
            rt = plan.runtime
            lines += [
                f"- state_model: `{plan.state_model}`",
                f"- oracles: `{', '.join(plan.oracles)}`",
                f"- contract_status: `{plan.contract_status.value}`",
                f"- oracle_contracts: `{json.dumps(plan.oracle_contracts, ensure_ascii=False)}`",
                f"- unimplemented_oracles: `{json.dumps(plan.metadata.get('unimplemented_oracles', []), ensure_ascii=False)}`",
                f"- external_threads: {rt.get('external_threads')}",
                f"- intra_op_threads: {rt.get('intra_op_threads')}",
                f"- inter_op_threads: {rt.get('inter_op_threads')}",
                f"- tf_data_parallel_calls: {rt.get('tf_data_parallel_calls')}",
                f"- trials: {plan.trials}",
                f"- timeout_seconds: {plan.timeout_seconds}",
                f"- case_source: `{plan.case_source}`",
            ]
        lines += ["", "## 3. Execution", ""]
        if observation:
            lines += [
                f"- run_id: `{observation.run_id}`",
                f"- mode: `{observation.mode}`",
                f"- trials: {observation.trials}",
                f"- baseline_run_id: `{observation.baseline_run_id}`",
                f"- test_run_id: `{observation.test_run_id}`",
                f"- config_id: `{observation.config_id}`",
                f"- process_exit_code: {observation.process_exit_code}",
                f"- process_signal: {observation.process_signal}",
                f"- observation_complete: {observation.observation_complete}",
                f"- task_progress: `{json.dumps(observation.task_progress, ensure_ascii=False)}`",
                f"- requested_runtime: `{json.dumps((observation.config or {}).get('runtime', {}), ensure_ascii=False)}`",
                f"- actual_runtime: `{json.dumps(observation.runtime, ensure_ascii=False)}`",
                f"- baseline_runtime: `{json.dumps(observation.baseline_runtime, ensure_ascii=False)}`",
            ]
        lines += ["", "## 4. Oracle Results", "", "| oracle | status | reason |", "|---|---|---|"]
        for e in result.oracle_results:
            lines.append(f"| {e.oracle_id} | **{e.status.value}** | {e.reason} |")
        lines += ["", "## 5. Evidence", ""]
        if result.failure_artifacts:
            for f in result.failure_artifacts:
                lines.append(f"- `{f}`")
        else:
            lines.append("- no failure artifacts")
        for e in result.oracle_results:
            if e.evidence and e.status in (OracleStatus.VIOLATION, OracleStatus.INCONCLUSIVE):
                lines.append(f"- {e.oracle_id}: `{json.dumps(e.evidence, ensure_ascii=False)[:2000]}`")
        lines += ["", "## 6. Reproduction", "", "- reproduction_rate: not measured (P3)", ""]
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def package_reproducer(self, result: OracleResult, path: str | Path) -> Path:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "oracle_result.json").write_text(
            json.dumps(to_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return p
