"""Fixed-template transformation from a fuzz_api_one harness to a concurrent harness."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .case_capture import CaseCapture
from .schemas import (
    ConcurrentHarnessArtifact,
    ConcurrentTestPlan,
    GenerationStatus,
    HarnessAnalysis,
    HarnessArtifact,
    ValidationStatus,
)


class HarnessTransformer:
    def __init__(self, output_dir: str | Path = "concurrent_oracle/outputs", template_path: Optional[str | Path] = None):
        self.output_dir = Path(output_dir).resolve()
        self.template_path = Path(template_path) if template_path else (
            Path(__file__).resolve().parent / "templates" / "concurrent_harness.py"
        )
        self.recipe_template_path = (
            Path(__file__).resolve().parent / "templates" / "recipe_harness.py"
        )

    def _target_patches(self, analysis: HarnessAnalysis) -> List[str]:
        patches: List[str] = []
        for call in analysis.target_calls:
            name = call.api_name
            if name.startswith("tf.") and name not in patches:
                patches.append(name)
        return patches

    def _transform_recipe(
        self,
        artifact: HarnessArtifact,
        plan: ConcurrentTestPlan,
        backend: str,
        gen_dir: Path,
        harness_out: Path,
        record_path: Path,
    ) -> ConcurrentHarnessArtifact:
        """Copy the recipe template; no target-call patching is required."""
        template = self.recipe_template_path.read_text(encoding="utf-8")
        harness_out.write_text(template, encoding="utf-8")
        record = {
            "plan_id": plan.plan_id,
            "artifact_id": artifact.artifact_id,
            "api_name": artifact.api_name,
            "source_harness": str(Path(artifact.harness_path).resolve()),
            "source_hash": artifact.harness_hash,
            "backend": backend,
            "recipe": plan.metadata.get("recipe"),
            "scenario": plan.scenario,
            "runtime": plan.runtime,
            "oracles": plan.oracles,
            "generation_status": "GENERATED",
            "generation_attempt": 1,
            "template": str(self.recipe_template_path),
            "warnings": [],
            "generated_at": time.time(),
        }
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False),
                               encoding="utf-8")
        return ConcurrentHarnessArtifact(
            artifact_id=artifact.artifact_id,
            source_harness=str(Path(artifact.harness_path).resolve()),
            concurrent_harness=str(harness_out),
            plan_id=plan.plan_id,
            generation_status=GenerationStatus.GENERATED,
            validation_status=ValidationStatus.PENDING,
            generation_attempt=1,
            oracle_bindings=list(plan.oracles),
            transform_record=record,
        )

    def transform(
        self,
        artifact: HarnessArtifact,
        analysis: HarnessAnalysis,
        plan: ConcurrentTestPlan,
        cases: Optional[List[Dict[str, Any]]] = None,
    ) -> ConcurrentHarnessArtifact:
        gen_dir = self.output_dir / "generated_harnesses" / plan.plan_id
        gen_dir.mkdir(parents=True, exist_ok=True)
        harness_out = gen_dir / "concurrent_harness.py"
        record_path = gen_dir / "transform_record.json"

        backend = str((plan.metadata or {}).get("backend", ""))
        if backend in {"tfdata_recipe", "callback_recipe"}:
            return self._transform_recipe(artifact, plan, backend, gen_dir,
                                          harness_out, record_path)

        patches = self._target_patches(analysis)
        if not patches:
            record = {
                "plan_id": plan.plan_id,
                "artifact_id": artifact.artifact_id,
                "source_harness": artifact.harness_path,
                "target_patches": [],
                "generation_status": "FAILED",
                "reason": "no patchable tf.* target call found by the analyzer",
            }
            record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return ConcurrentHarnessArtifact(
                artifact_id=artifact.artifact_id,
                source_harness=artifact.harness_path,
                concurrent_harness="",
                plan_id=plan.plan_id,
                generation_status=GenerationStatus.FAILED,
                validation_status=ValidationStatus.FAILED,
                oracle_bindings=list(plan.oracles),
                transform_record=record,
            )

        capture = CaseCapture(self.output_dir)
        case_source = "argfuzz_corpus" if cases else "deterministic_defaults"
        cases_path = capture.write_cases(plan.plan_id, cases=cases, metadata={
            "source": case_source,
            "original_harness": artifact.harness_path,
            "api_name": artifact.api_name,
        })

        template = self.template_path.read_text(encoding="utf-8")
        template = template.replace(
            "ORIGINAL_HARNESS = __ORIGINAL_HARNESS__",
            f"ORIGINAL_HARNESS = {json.dumps(str(Path(artifact.harness_path).resolve()))}",
        )
        template = template.replace(
            "TARGET_PATCHES = __TARGET_PATCHES__",
            f"TARGET_PATCHES = {json.dumps(patches)}",
        )
        # The embedded defaults are only a fallback; cases.json is authoritative.
        default_b64 = [c["data_b64"] for c in capture.default_cases()]
        template = template.replace(
            "DEFAULT_CASES = __DEFAULT_CASES__",
            f"DEFAULT_CASES = {json.dumps(default_b64)}",
        )
        harness_out.write_text(template, encoding="utf-8")

        record = {
            "plan_id": plan.plan_id,
            "artifact_id": artifact.artifact_id,
            "api_name": artifact.api_name,
            "source_harness": str(Path(artifact.harness_path).resolve()),
            "source_hash": artifact.harness_hash,
            "target_patches": patches,
            "cases_path": str(cases_path),
            "case_source": case_source,
            "scenario": plan.scenario,
            "runtime": plan.runtime,
            "oracles": plan.oracles,
            "generation_status": "GENERATED",
            "generation_attempt": 1,
            "template": str(self.template_path),
            "warnings": list(analysis.warnings),
            "generated_at": time.time(),
        }
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        return ConcurrentHarnessArtifact(
            artifact_id=artifact.artifact_id,
            source_harness=str(Path(artifact.harness_path).resolve()),
            concurrent_harness=str(harness_out),
            plan_id=plan.plan_id,
            generation_status=GenerationStatus.GENERATED,
            validation_status=ValidationStatus.PENDING,
            generation_attempt=1,
            oracle_bindings=list(plan.oracles),
            transform_record=record,
        )
