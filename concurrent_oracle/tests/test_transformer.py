import ast
import json
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.api_suitability_analyzer import APISuitabilityAnalyzer
from concurrent_oracle.contract_registry import ContractRegistry
from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader
from concurrent_oracle.scenario_generator import ScenarioGenerator
from concurrent_oracle.suitability_rules import SuitabilityRules
from concurrent_oracle.transformer import HarnessTransformer

REAL = Path("/root/fuzz_output_experiment/llm.tf.math.log1p.py")
CONTRACTS = Path("/root/concurrent_oracle/contracts/tensorflow")


class TestTransformer(unittest.TestCase):
    def test_generate_concurrent_harness(self):
        sha_before = HarnessLoader().calculate_hash(REAL)
        artifact = HarnessLoader().load(REAL)
        analysis = HarnessAnalyzer().analyze(artifact)
        registry = ContractRegistry(CONTRACTS)
        report = APISuitabilityAnalyzer(SuitabilityRules(), None, registry, {}).analyze(artifact, analysis)
        plan = ScenarioGenerator(contract_registry=registry).generate(report, api_name=artifact.api_name)[0]
        with tempfile.TemporaryDirectory() as td:
            art = HarnessTransformer(td).transform(artifact, analysis, plan)
            self.assertTrue(art.concurrent_harness)
            path = Path(art.concurrent_harness)
            self.assertTrue(path.exists())
            ast.parse(path.read_text(encoding="utf-8"))
            text = path.read_text(encoding="utf-8")
            self.assertIn("TARGET_PATCHES = [\"tf.math.log1p\", \"tf.raw_ops.Log1p\"]", text)
            self.assertIn(str(REAL.resolve()), text)
            cases = Path(td) / "captured_cases" / plan.plan_id / "cases.json"
            self.assertTrue(cases.exists())
            data = json.loads(cases.read_text(encoding="utf-8"))
            self.assertEqual(len(data["cases"]), 8)
            record = Path(td) / "generated_harnesses" / plan.plan_id / "transform_record.json"
            self.assertTrue(record.exists())
        # original harness must not be modified
        self.assertEqual(HarnessLoader().calculate_hash(REAL), sha_before)


if __name__ == "__main__":
    unittest.main()
