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

REAL = Path(__file__).resolve().parent / "fixtures" / "llm.tf.math.log1p.py"
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "tensorflow"


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
            self.assertIn('TARGET_PATCHES = ["tf.math.log1p"]', text)
            self.assertIn(str(REAL.resolve()), text)
            cases = Path(td) / "captured_cases" / plan.plan_id / "cases.json"
            self.assertTrue(cases.exists())
            data = json.loads(cases.read_text(encoding="utf-8"))
            self.assertEqual(len(data["cases"]), 8)
            record = Path(td) / "generated_harnesses" / plan.plan_id / "transform_record.json"
            self.assertTrue(record.exists())

            # The template must serialize 0-d / scalar / complex / string values.
            import importlib.util
            import numpy as np
            spec = importlib.util.spec_from_file_location("co_generated_test", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertEqual(mod._jsonable(np.array(1.0)), 1.0)
            self.assertEqual(mod._jsonable(np.array([1, 2])), [1, 2])
            self.assertEqual(mod._jsonable(np.array(["a", "b"])), ["a", "b"])
            self.assertEqual(mod._jsonable(np.array([], dtype=float)), [])
            self.assertEqual(mod._jsonable(np.array(True)), True)
            self.assertEqual(mod._jsonable(np.array([1 + 2j])),
                             [{"real": 1.0, "imag": 2.0}])
            self.assertEqual(mod._array_sha256(np.array(["a", "b"])),
                             mod._array_sha256(np.array(["a", "b"])))
            self.assertNotEqual(mod._array_sha256(np.array(["a", "b"])),
                                mod._array_sha256(np.array(["a", "c"])))
        # original harness must not be modified
        self.assertEqual(HarnessLoader().calculate_hash(REAL), sha_before)


if __name__ == "__main__":
    unittest.main()
