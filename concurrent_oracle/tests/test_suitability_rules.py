import unittest
from pathlib import Path

from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader
from concurrent_oracle.suitability_rules import SuitabilityRules

REAL = Path("/root/fuzz_output_experiment/llm.tf.math.log1p.py")


class TestSuitabilityRules(unittest.TestCase):
    def setUp(self):
        self.loader = HarnessLoader()
        self.analyzer = HarnessAnalyzer()
        self.rules = SuitabilityRules()

    def test_log1p_rules(self):
        artifact = self.loader.load(REAL)
        analysis = self.analyzer.analyze(artifact)
        result = self.rules.analyze(artifact, analysis)
        self.assertIn("PURE_TENSOR_OP", result["api_categories"])
        self.assertIn("independent_api_calls", [s["scenario_id"] for s in result["candidate_scenarios"]])
        self.assertIn("shared_readonly_input", [s["scenario_id"] for s in result["candidate_scenarios"]])
        self.assertIn("O2", result["rule_candidates"])
        self.assertIn("O3", result["rule_candidates"])
        for s in result["candidate_scenarios"]:
            self.assertIn(s["state_model"], ("INDEPENDENT_INPUTS", "SHARED_READONLY"))

    def test_no_target_call_no_ready_scenario(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as td:
            p = Path(td) / "llm.tf.none.py"
            p.write_text("import tensorflow as tf\nx = 1\n", encoding="utf-8")
            artifact = self.loader.load(p, api_name="tf.none")
            analysis = self.analyzer.analyze(artifact)
            result = self.rules.analyze(artifact, analysis)
            self.assertIn("UNKNOWN", result["api_categories"])


if __name__ == "__main__":
    unittest.main()
