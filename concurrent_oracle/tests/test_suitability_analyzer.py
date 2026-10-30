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

REAL = Path(__file__).resolve().parent / "fixtures" / "llm.tf.math.log1p.py"
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "tensorflow"


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
        # First pass covers each scenario once, before expanding configurations.
        self.assertEqual([p.scenario for p in plans],
                         ["independent_api_calls", "shared_readonly_input",
                          "independent_api_calls", "shared_readonly_input"])
        self.assertEqual([p.runtime["external_threads"] for p in plans], [2, 2, 4, 4])
        for p in plans:
            self.assertIn("O3", p.oracles)
            if p.scenario == "independent_api_calls":
                self.assertIn("O2", p.oracles)
            self.assertEqual(p.trials, 10)
            self.assertEqual(p.timeout_seconds, 60)
            self.assertTrue(p.plan_id.startswith(self.artifact.artifact_id))

    def test_per_scenario_contracts(self):
        from concurrent_oracle.contract_registry import ContractRegistry
        from concurrent_oracle.schemas import CandidateScenario, SuitabilityReport
        reg = ContractRegistry()
        reg.register_contract({"api_name": "tf.test.mixed", "scenario": "independent_api_calls",
                               "status": "SUPPORTED", "source": "test"})
        reg.register_contract({"api_name": "tf.test.mixed", "scenario": "shared_readonly_input",
                               "status": "UNKNOWN", "source": "test"})
        report = SuitabilityReport(
            artifact_id="mixed", decision=Decision.READY, contract_status=ContractStatus.SUPPORTED,
            candidate_scenarios=[
                CandidateScenario("independent_api_calls", ["O2", "O3"], "INDEPENDENT_INPUTS", "HIGH", "x"),
                CandidateScenario("shared_readonly_input", ["O3", "O4"], "SHARED_READONLY", "MEDIUM", "y"),
            ],
        )
        plans = ScenarioGenerator(config=ModuleConfig(), contract_registry=reg).generate(
            report, api_name="tf.test.mixed")
        by = {(p.scenario, p.runtime["config_id"]): p for p in plans}
        self.assertEqual(by[("independent_api_calls", "C03")].contract_status, ContractStatus.SUPPORTED)
        self.assertEqual(by[("shared_readonly_input", "C03")].contract_status, ContractStatus.UNKNOWN)
        self.assertEqual(by[("shared_readonly_input", "C03")].oracle_contracts.get("O3"), "UNKNOWN")

    def test_runtime_config_filtering(self):
        cfg = ModuleConfig()
        cfg.runtime.external_threads = [1, 2]
        cfg.runtime.intra_op_threads = [1]
        cfg.runtime.inter_op_threads = [1]
        cfg.runtime.tf_data_parallel_calls = [1]
        gen = ScenarioGenerator(config=cfg, contract_registry=self.registry)
        configs = gen.generate_runtime_configs("independent_api_calls")
        self.assertEqual([c["config_id"] for c in configs], ["C03"])
        self.assertNotIn("C04", [c["config_id"] for c in configs])

    def test_unimplemented_scenario_skipped_with_reason(self):
        from concurrent_oracle.schemas import CandidateScenario, SuitabilityReport
        report = SuitabilityReport(
            artifact_id="a", decision=Decision.READY, contract_status=ContractStatus.SUPPORTED,
            candidate_scenarios=[
                CandidateScenario("shared_resource_access", ["O3"], "SHARED_MUTABLE", "MEDIUM", "x"),
            ],
        )
        cfg = ModuleConfig()
        gen = ScenarioGenerator(config=cfg, contract_registry=self.registry)
        plans = gen.generate(report, api_name=self.artifact.api_name)
        self.assertEqual(plans, [])
        self.assertIn("shared_resource_access", gen.last_skipped)
        self.assertIn("P1", gen.last_skipped["shared_resource_access"])

    def test_conditional_for_unknown_contract(self):
        report = self.analyzer.analyze(self.artifact, self.analysis)
        # A registry without tf.math.log1p must produce CONDITIONAL, not READY.
        empty = APISuitabilityAnalyzer(self.rules, None, ContractRegistry(), {})
        r2 = empty.analyze(self.artifact, self.analysis)
        self.assertEqual(r2.decision, Decision.CONDITIONAL)
        self.assertEqual(r2.contract_status, ContractStatus.UNKNOWN)


if __name__ == "__main__":
    unittest.main()
