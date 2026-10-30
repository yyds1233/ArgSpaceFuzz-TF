"""Command-line interface for concurrent_oracle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pipeline import ConcurrentOraclePipeline
from .schemas import (
    ConcurrentHarnessArtifact,
    ConcurrentTestPlan,
    OracleResult,
    read_json,
    to_jsonable,
)


def _load_plan(pipeline: ConcurrentOraclePipeline, spec: str) -> ConcurrentTestPlan:
    p = Path(spec)
    if p.exists():
        return ConcurrentTestPlan.from_dict(read_json(p))
    return pipeline.load_plan(spec)


def _load_harness_artifact(pipeline: ConcurrentOraclePipeline, plan: ConcurrentTestPlan) -> ConcurrentHarnessArtifact:
    p = pipeline.gen_dir(plan.plan_id) / "concurrent_harness_artifact.json"
    if p.exists():
        return ConcurrentHarnessArtifact.from_dict(read_json(p))
    return None  # type: ignore[return-value]


def cmd_analyze(args) -> int:
    pipeline = ConcurrentOraclePipeline(args.config, args.output_dir)
    paths: List[Path] = []
    if args.harness:
        paths = [Path(args.harness)]
    elif args.harness_dir:
        d = Path(args.harness_dir)
        paths = sorted(p for p in d.glob("*.py") if not p.name.startswith("__"))
    else:
        raise SystemExit("analyze needs --harness or --harness-dir")
    rows = []
    for hp in paths:
        artifact, analysis, report, plans = pipeline.analyze(hp, psr_path=args.psr)
        rows.append({
            "harness": str(hp),
            "artifact_id": artifact.artifact_id,
            "api_name": artifact.api_name,
            "decision": report.decision.value,
            "contract_status": report.contract_status.value,
            "scenarios": [s.scenario_id for s in report.candidate_scenarios],
            "plans": [p.plan_id for p in plans],
            "warnings": analysis.warnings,
        })
        print(f"[analyze] {artifact.api_name} decision={report.decision.value} "
              f"scenarios={[s.scenario_id for s in report.candidate_scenarios]}", flush=True)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def cmd_generate(args) -> int:
    pipeline = ConcurrentOraclePipeline(args.config, args.output_dir)
    plan = _load_plan(pipeline, args.plan)
    artifact = pipeline.load_artifact(plan.artifact_id)
    analysis = pipeline.load_analysis(plan.artifact_id)
    art = pipeline.generate(artifact, analysis, plan)
    result: Dict[str, Any] = {"plan_id": plan.plan_id, "artifact": to_jsonable(art)}
    if not args.no_validate:
        result["validation"] = pipeline.validate(art, plan, run_baseline=not args.no_serial_baseline)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args) -> int:
    pipeline = ConcurrentOraclePipeline(args.config, args.output_dir)
    plan = _load_plan(pipeline, args.plan)
    artifact = pipeline.load_artifact(plan.artifact_id)
    analysis = pipeline.load_analysis(plan.artifact_id)
    art = _load_harness_artifact(pipeline, plan)
    if art is None:
        art = pipeline.generate(artifact, analysis, plan)
    if not args.skip_validate:
        val = pipeline.validate(art, plan, run_baseline=not args.no_serial_baseline)
        if not val.get("ok"):
            print(json.dumps({"plan_id": plan.plan_id, "stage": "validate",
                              "reason": val.get("serial_baseline", {}).get("reason") or "validation failed"},
                             indent=2, ensure_ascii=False))
            return 1
    obs = pipeline.run(art, plan)
    result = pipeline.evaluate(obs, plan)
    report = pipeline.report(artifact, plan, obs, result)
    print(json.dumps({
        "plan_id": plan.plan_id,
        "execution_status": obs.execution_status.value,
        "task_progress": obs.task_progress,
        "oracle_results": [
            {"oracle_id": e.oracle_id, "status": e.status.value, "reason": e.reason}
            for e in result.oracle_results
        ],
        "report": str(report),
    }, indent=2, ensure_ascii=False))
    return 0


def cmd_pipeline(args) -> int:
    pipeline = ConcurrentOraclePipeline(args.config, args.output_dir)
    paths: List[Path] = []
    if args.harness:
        paths = [Path(args.harness)]
    elif args.harness_dir:
        d = Path(args.harness_dir)
        paths = sorted(p for p in d.glob("*.py") if not p.name.startswith("__"))
    else:
        raise SystemExit("pipeline needs --harness or --harness-dir")
    summaries = []
    for hp in paths:
        print(f"[pipeline] harness {hp}", flush=True)
        summary = pipeline.run_harness(
            hp, psr_path=args.psr,
            max_plans=args.max_plans if args.max_plans > 0 else 10**9,
            run_baseline=not args.no_serial_baseline,
            allow_unknown_contract=args.allow_unknown_contract,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_report(args) -> int:
    pipeline = ConcurrentOraclePipeline(args.config, args.output_dir)
    result = OracleResult.from_dict(read_json(args.result))
    plan = None
    artifact = None
    try:
        plan = pipeline.load_plan(result.plan_id)
        artifact = pipeline.load_artifact(plan.artifact_id)
    except Exception:
        pass
    out = pipeline.reporter.write_markdown(
        result, pipeline.report_path(result.plan_id),
        artifact=artifact, plan=plan, observation=None,
    )
    print(str(out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="TensorFlow CPU concurrent oracle")
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", default=None)
        p.add_argument("--output-dir", default=None)

    p = sub.add_parser("analyze")
    common(p)
    p.add_argument("--harness", default=None)
    p.add_argument("--harness-dir", default=None)
    p.add_argument("--psr", default=None)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("generate")
    common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--no-serial-baseline", action="store_true")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--plan", required=True)
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--no-serial-baseline", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("pipeline")
    common(p)
    p.add_argument("--harness", default=None)
    p.add_argument("--harness-dir", default=None)
    p.add_argument("--psr", default=None)
    p.add_argument("--max-plans", type=int, default=0,
                   help="0 means all plans; otherwise the first N plans")
    p.add_argument("--no-serial-baseline", action="store_true")
    p.add_argument("--allow-unknown-contract", action="store_true")
    p.set_defaults(func=cmd_pipeline)

    p = sub.add_parser("report")
    common(p)
    p.add_argument("--result", required=True)
    p.set_defaults(func=cmd_report)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
