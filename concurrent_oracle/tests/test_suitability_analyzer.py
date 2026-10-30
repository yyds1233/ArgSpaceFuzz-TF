import unittest
from pathlib import Path

from concurrent_oracle.api_suitability_analyzer import APISuitabilityAnalyzer
from concurrent_oracle.config import ModuleConfig
from concurrent_oracle.contract_registry import ContractRegistry
from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader
from concurrent_oracle.scenario_generator import ScenarioGenerator
from concurrent_oracle.schemas import Decision, ContractStatus
from concurrent_oracle.suitability_rules import SuitabilityRules

REAL = Path("/root/fuzz_output_experiment/llm.tf.math.log1p.py")
CONTRACTS = Path("/root/concurrent_oracle/contracts/tensorflow")


class TestSuitabilityAndPlans(unittest.TestCase):
    def setUp(self):
        self.artifact = HarnessLoader().load(REAL)
        self.analysis = HarnessAnalyzer().analyze(self.artifact)
        self.registry = ContractRegistry(CONTRACTS)
        self.rules = SuitabilityRules()
        self.analyzer = APISuitabilityAnalyzer(
            rules=self.rules,
            llm_client=None,
            contract_registry=self.registry,
            config={"model_config": {"model": "none"}},
        )

    def test_report_ready_with_supported_contract(self):
        report = self.analyzer.analyze(self.artifact, self.analysis)
        self.assertEqual(report.decision, Decision.READY)
        self.assertEqual(report.contract_status, ContractStatus.SUPPORTED)
        self.assertTrue(report.candidate_scenarios)
        self.assertIn("O2", report.rule_candidates)
        self.assertIn("PURE_TENSOR_OP", report.api_categories)

    def test_generate_plans(self):
        report = self.analyzer.analyze(self.artifact, self.analysis)
        gen = ScenarioGenerator(config=ModuleConfig(), contract_registry=self.registry)
        plans = gen.generate(report, api_name=self.artifact.api_name)
        self.assertEqual(len(plans), 4)
        self.assertEqual(plans[0].scenario, "independent_api_calls")
        self.assertEqual(plans[0].runtime["external_threads"], 2)
        self.assertEqual(plans[1].runtime["external_threads"], 4)
        for p in plans:
            self.assertIn("O3", p.oracles)
            if p.scenario == "independent_api_calls":
                self.assertIn("O2", p.oracles)
            self.assertEqual(p.trials, 10)
            self.assertEqual(p.timeout_seconds, 60)
            self.assertTrue(p.plan_id.startswith(self.artifact.artifact_id))

    def test_conditional_for_unknown_contract(self):
        report = self.analyzer.analyze(self.artifact, self.analysis)
        # A registry without tf.math.log1p must produce CONDITIONAL, not READY.
        empty = APISuitabilityAnalyzer(self.rules, None, ContractRegistry(), {})
        r2 = empty.analyze(self.artifact, self.analysis)
        self.assertEqual(r2.decision, Decision.CONDITIONAL)
        self.assertEqual(r2.contract_status, ContractStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
