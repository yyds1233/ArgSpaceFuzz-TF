import os
import sys
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.api_suitability_analyzer import APISuitabilityAnalyzer
from concurrent_oracle.contract_registry import ContractRegistry
from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader
from concurrent_oracle.harness_validator import HarnessValidator
from concurrent_oracle.scenario_generator import ScenarioGenerator
from concurrent_oracle.suitability_rules import SuitabilityRules
from concurrent_oracle.transformer import HarnessTransformer

REAL = Path(__file__).resolve().parent / "fixtures" / "llm.tf.math.log1p.py"
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "tensorflow"


def build(tmp: str):
    artifact = HarnessLoader().load(REAL)
    analysis = HarnessAnalyzer().analyze(artifact)
    registry = ContractRegistry(CONTRACTS)
    report = APISuitabilityAnalyzer(SuitabilityRules(), None, registry, {}).analyze(artifact, analysis)
    plan = ScenarioGenerator(contract_registry=registry).generate(report, api_name=artifact.api_name)[0]
    art = HarnessTransformer(tmp).transform(artifact, analysis, plan)
    return artifact, analysis, plan, art


class TestHarnessValidator(unittest.TestCase):
    def test_static_checks(self):
        with tempfile.TemporaryDirectory() as td:
            artifact, analysis, plan, art = build(td)
            v = HarnessValidator(td, python_path=sys.executable)
            self.assertTrue(v.validate_syntax(art.concurrent_harness)["ok"])
            self.assertTrue(v.validate_imports(art.concurrent_harness)["ok"])
            target = v.validate_target_call(art.concurrent_harness, artifact.api_name)
            self.assertTrue(target["ok"], target)
            worker = v.validate_worker_structure(art.concurrent_harness)
            self.assertTrue(worker["ok"], worker)

    def test_broken_import(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "broken.py"
            p.write_text("this is not python !!!\n", encoding="utf-8")
            v = HarnessValidator(td, python_path=sys.executable)
            self.assertFalse(v.validate_syntax(p)["ok"])

    @unittest.skipUnless(os.environ.get("CO_RUN_TF_TESTS") == "1", "set CO_RUN_TF_TESTS=1")
    def test_serial_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            artifact, analysis, plan, art = build(td)
            v = HarnessValidator(td, python_path=sys.executable)
            r = v.run_serial_baseline(art.concurrent_harness, plan, td, timeout=120)
            self.assertTrue(r["ok"], r.get("reason"))
            self.assertGreaterEqual(len(r.get("completed_cases", [])), 1)


if __name__ == "__main__":
    unittest.main()
