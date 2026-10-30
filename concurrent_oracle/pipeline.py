"""End-to-end orchestration for the TensorFlow concurrent oracle (P0)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api_suitability_analyzer import APISuitabilityAnalyzer
from .config import ModuleConfig, load_config
from .contract_registry import ContractRegistry
from .harness_analyzer import HarnessAnalyzer
from .harness_loader import HarnessLoader
from .harness_validator import HarnessValidator
from .oracle_engine import OracleEngine
from .reporter import Reporter
from .runner import ConcurrentRunner
from .scenario_generator import ScenarioGenerator
from .schemas import (
    ConcurrentHarnessArtifact,
    ConcurrentTestPlan,
    ContractStatus,
    Decision,
    ExecutionObservation,
    HarnessAnalysis,
    HarnessArtifact,
    OracleResult,
    SuitabilityReport,
    read_json,
    to_jsonable,
    write_json,
)
from .suitability_rules import SuitabilityRules
from .transformer import HarnessTransformer

MODULE_ROOT = Path(__file__).resolve().parent.parent


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (MODULE_ROOT / p).resolve()


class ConcurrentOraclePipeline:
    def __init__(
        self,
        config: Optional[ModuleConfig | str | Path] = None,
        output_dir: Optional[str | Path] = None,
    ):
        if isinstance(config, ModuleConfig):
            self.config = config
        else:
            self.config = load_config(config)
        self.output_dir = _resolve(output_dir) if output_dir else _resolve(self.config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = HarnessLoader()
        self.analyzer = HarnessAnalyzer()
        self.rules = SuitabilityRules()
        self.contract_registry = ContractRegistry(_resolve(self.config.paths.contract_dir))
        self.suitability = APISuitabilityAnalyzer(
            rules=self.rules,
            llm_client=None,
            contract_registry=self.contract_registry,
            config={"model_config": {}},
        )
        self.scenario_generator = ScenarioGenerator(
            config=self.config,
            contract_registry=self.contract_registry,
        )
        self.transformer = HarnessTransformer(self.output_dir)
        self.validator = HarnessValidator(self.output_dir, python_path=sys.executable)
        self.runner = ConcurrentRunner(self.output_dir, python_path=sys.executable)
        self.oracle_engine = OracleEngine()
        self.reporter = Reporter()

    # ── persistence paths ──────────────────────────────────────

    def artifact_path(self, artifact_id: str) -> Path:
        return self.output_dir / "artifacts" / f"{artifact_id}.json"

    def analysis_path(self, artifact_id: str) -> Path:
        return self.output_dir / "analysis" / f"{artifact_id}_analysis.json"

    def suitability_path(self, artifact_id: str) -> Path:
        return self.output_dir / "suitability" / f"{artifact_id}_suitability.json"

    def plan_path(self, plan_id: str) -> Path:
        return self.output_dir / "plans" / f"{plan_id}.json"

    def gen_dir(self, plan_id: str) -> Path:
        return self.output_dir / "generated_harnesses" / plan_id

    def execution_dir(self, plan_id: str, config_id: str) -> Path:
        return self.output_dir / "executions" / plan_id / config_id

    def oracle_result_path(self, plan_id: str) -> Path:
        return self.output_dir / "oracle_results" / f"{plan_id}.json"

    def report_path(self, plan_id: str) -> Path:
        return self.output_dir / "reports" / f"{plan_id}_report.md"

    # ── stages ─────────────────────────────────────────────────

    def analyze(self, harness_path: str | Path, psr_path: Optional[str | Path] = None):
        artifact = self.loader.load(harness_path, psr_path=psr_path)
        analysis = self.analyzer.analyze(artifact)
        report = self.suitability.analyze(artifact, analysis)
        plans = self.scenario_generator.generate(report, api_name=artifact.api_name)
        self.last_skipped = dict(self.scenario_generator.last_skipped)

        write_json(self.artifact_path(artifact.artifact_id), artifact)
        write_json(self.analysis_path(artifact.artifact_id), analysis)
        write_json(self.suitability_path(artifact.artifact_id), report)
        for plan in plans:
            write_json(self.plan_path(plan.plan_id), plan)
        return artifact, analysis, report, plans

    def generate(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
        plan: ConcurrentTestPlan,
    ) -> ConcurrentHarnessArtifact:
        art = self.transformer.transform(artifact, analysis, plan)
        gen_dir = self.gen_dir(plan.plan_id)
        write_json(gen_dir / "concurrent_harness_artifact.json", art)
        return art

    def validate(
        self,
        art: ConcurrentHarnessArtifact,
        plan: ConcurrentTestPlan,
        run_baseline: bool = True,
    ) -> Dict[str, Any]:
        result = self.validator.validate(art, plan, self.output_dir, run_baseline=run_baseline)
        write_json(self.gen_dir(plan.plan_id) / "validation.json", result)
        write_json(self.gen_dir(plan.plan_id) / "concurrent_harness_artifact.json", art)
        return result

    def run(self, art: ConcurrentHarnessArtifact, plan: ConcurrentTestPlan) -> ExecutionObservation:
        return self.runner.run(art, plan)

    def evaluate(self, observation: ExecutionObservation, plan: ConcurrentTestPlan) -> OracleResult:
        result = self.oracle_engine.evaluate(
            observation, plan,
            context={"comparison": to_jsonable(self.config.comparison)},
        )
        write_json(self.oracle_result_path(plan.plan_id), result)
        return result

    def report(
        self,
        artifact: HarnessArtifact,
        plan: ConcurrentTestPlan,
        observation: ExecutionObservation,
        result: OracleResult,
    ) -> Path:
        return self.reporter.write_markdown(
            result, self.report_path(plan.plan_id),
            artifact=artifact, plan=plan, observation=observation,
        )

    # ── end-to-end ─────────────────────────────────────────────

    def run_harness(
        self,
        harness_path: str | Path,
        psr_path: Optional[str | Path] = None,
        max_plans: int = 1,
        run_baseline: bool = True,
        allow_unknown_contract: bool = False,
    ) -> Dict[str, Any]:
        artifact, analysis, report, plans = self.analyze(harness_path, psr_path=psr_path)
        summary: Dict[str, Any] = {
            "artifact": to_jsonable(artifact),
            "decision": report.decision.value,
            "contract_status": report.contract_status.value,
            "scenarios": [s.scenario_id for s in report.candidate_scenarios],
            "plans": [to_jsonable(p) for p in plans],
            "skipped_scenarios": getattr(self, "last_skipped", {}),
            "contract_load_errors": list(self.contract_registry.load_errors),
            "results": [],
            "errors": [],
            "output_dir": str(self.output_dir),
        }
        if report.decision != Decision.READY:
            summary["skipped"] = f"decision is {report.decision.value}; no plan executed"
            return summary
        if not plans:
            summary["skipped"] = "no plans generated"
            return summary

        for plan in plans[:max_plans]:
            if plan.contract_status == ContractStatus.UNSUPPORTED:
                summary.setdefault("skipped_plans", []).append(
                    {"plan_id": plan.plan_id, "reason": "contract UNSUPPORTED"})
                continue
            if plan.contract_status == ContractStatus.UNKNOWN and not allow_unknown_contract:
                summary.setdefault("skipped_plans", []).append(
                    {"plan_id": plan.plan_id, "reason": "contract UNKNOWN (strict mode)"})
                continue
            print(f"[pipeline] plan {plan.plan_id} scenario={plan.scenario} "
                  f"config={plan.runtime.get('config_id')} "
                  f"contract={plan.contract_status.value}", flush=True)
            try:
                art = self.generate(artifact, analysis, plan)
                if not art.concurrent_harness:
                    summary["errors"].append({"plan_id": plan.plan_id,
                                              "stage": "generate", "reason": "no harness generated"})
                    continue
                capture = self.runner.capture_cases(art, plan)
                summary.setdefault("captures", []).append({"plan_id": plan.plan_id, **capture})
                if capture.get("status") == "FALLBACK" and not capture.get("cases_with_target_calls"):
                    summary["errors"].append({"plan_id": plan.plan_id, "stage": "capture",
                                              "reason": "no target call could be captured; "
                                                        "bytes fallback is disabled in strict P0"})
                    continue
                if capture.get("unsupported_target_cases"):
                    summary["errors"].append({"plan_id": plan.plan_id, "stage": "capture",
                                              "reason": f"{capture['unsupported_target_cases']} "
                                                        f"case(s) use unsupported target arguments"})
                    continue
                # Static validation only here; runner.run performs the isolated
                # baseline and concurrent processes.
                val = self.validate(art, plan, run_baseline=False)
                if not val.get("ok"):
                    summary["errors"].append({"plan_id": plan.plan_id, "stage": "validate",
                                              "reason": "static validation failed"})
                    continue
                obs = self.run(art, plan)
                if obs.execution_status.value not in ("COMPLETED", "PARTIAL"):
                    summary["errors"].append({
                        "plan_id": plan.plan_id, "stage": "run",
                        "reason": obs.execution_status.value, "run_id": obs.run_id,
                        "baseline_run_id": obs.baseline_run_id,
                    })
                result = self.evaluate(obs, plan)
                report_path = self.report(artifact, plan, obs, result)
                summary["results"].append({
                    "plan_id": plan.plan_id,
                    "scenario": plan.scenario,
                    "config_id": plan.runtime.get("config_id"),
                    "run_id": obs.run_id,
                    "baseline_run_id": obs.baseline_run_id,
                    "test_run_id": obs.test_run_id,
                    "trials": obs.trials,
                    "execution_status": obs.execution_status.value,
                    "oracle_results": [
                        {"oracle_id": e.oracle_id, "status": e.status.value, "reason": e.reason}
                        for e in result.oracle_results
                    ],
                    "report": str(report_path),
                    "oracle_result": str(self.oracle_result_path(plan.plan_id)),
                })
            except Exception as exc:
                summary["errors"].append({"plan_id": plan.plan_id, "stage": "pipeline",
                                          "reason": f"{type(exc).__name__}: {exc}"})
        return summary

    # ── loaders for individual CLI stages ──────────────────────

    def load_artifact(self, artifact_id: str) -> HarnessArtifact:
        return HarnessArtifact.from_dict(read_json(self.artifact_path(artifact_id)))

    def load_analysis(self, artifact_id: str) -> HarnessAnalysis:
        return HarnessAnalysis.from_dict(read_json(self.analysis_path(artifact_id)))

    def load_plan(self, plan_id: str) -> ConcurrentTestPlan:
        return ConcurrentTestPlan.from_dict(read_json(self.plan_path(plan_id)))

    def load_artifact_for_plan(self, plan: ConcurrentTestPlan) -> HarnessArtifact:
        return self.load_artifact(plan.artifact_id)

    def load_analysis_for_plan(self, plan: ConcurrentTestPlan) -> HarnessAnalysis:
        return self.load_analysis(plan.artifact_id)
