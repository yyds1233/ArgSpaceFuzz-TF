import unittest
from pathlib import Path

from concurrent_oracle.api_suitability_analyzer import APISuitabilityAnalyzer
from concurrent_oracle.config import ModuleConfig
from concurrent_oracle.contract_registry import ContractRegistry
from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader
from concurrent_oracle.scenario_generator import ScenarioGenerator
from concurrent_oracle.suitability_rules import SuitabilityRules

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "tensorflow"


def plans_for(fixture: str, enabled, runtime=None):
    artifact = HarnessLoader().load(FIXTURES / fixture)
    analysis = HarnessAnalyzer().analyze(artifact)
    registry = ContractRegistry(CONTRACTS)
    rules = SuitabilityRules({"enabled_oracles": list(enabled)})
    report = APISuitabilityAnalyzer(
        rules=rules, llm_client=None, contract_registry=registry, config={}
    ).analyze(artifact, analysis)
    cfg = ModuleConfig()
    if runtime:
        for key, value in runtime.items():
            setattr(cfg.runtime, key, value)
    cfg.oracle_enabled = list(enabled)
    plans = ScenarioGenerator(config=cfg, contract_registry=registry).generate(
        report, api_name=artifact.api_name
    )
    return artifact, report, plans


class TestExtendedScenarios(unittest.TestCase):
    def test_pointer_api_gets_o1_o4_o5(self):
        artifact, report, plans = plans_for(
            "llm.tf.math.log1p.py", ["O1", "O2", "O3", "O4", "O5"],
            runtime={"external_threads": [1, 2], "intra_op_threads": [1],
                     "inter_op_threads": [1], "tf_data_parallel_calls": [1],
                     "deterministic": [True]},
        )
        scenarios = {p.scenario for p in plans}
        self.assertIn("independent_api_calls", scenarios)
        self.assertIn("invalid_input_cpu", scenarios)
        self.assertIn("exception_recovery", scenarios)
        by = {p.scenario: p for p in plans}
        self.assertIn("O4", by["independent_api_calls"].oracles)
        self.assertEqual(by["invalid_input_cpu"].metadata["backend"], "invalid_input_replay")
        self.assertEqual(by["exception_recovery"].metadata["backend"], "exception_recovery_replay")

    def test_tfdata_fixture_gets_recipe(self):
        artifact, report, plans = plans_for(
            "llm.tf.data.Dataset.map.py", ["O6"],
            runtime={"external_threads": [1], "intra_op_threads": [1],
                     "inter_op_threads": [1], "tf_data_parallel_calls": [1, 4],
                     "deterministic": [True]},
        )
        self.assertTrue(plans)
        for p in plans:
            if p.scenario.startswith("tfdata_"):
                self.assertEqual(p.metadata["backend"], "tfdata_recipe")
                self.assertIn("kind", p.metadata["recipe"])

    def test_callback_fixture_gets_recipe(self):
        artifact, report, plans = plans_for(
            "llm.tf.py_function.py", ["O7"],
            runtime={"external_threads": [1, 2], "intra_op_threads": [1],
                     "inter_op_threads": [1], "tf_data_parallel_calls": [1],
                     "deterministic": [True]},
        )
        callback_plans = [p for p in plans if p.scenario == "python_callback_lifecycle"]
        self.assertTrue(callback_plans)
        for p in callback_plans:
            self.assertEqual(p.metadata["backend"], "callback_recipe")
            self.assertEqual(p.metadata["recipe"]["kind"], "python_callback_lifecycle")


if __name__ == "__main__":
    unittest.main()
