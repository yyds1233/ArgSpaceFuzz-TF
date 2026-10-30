import unittest
from pathlib import Path

from concurrent_oracle.contract_registry import ContractRegistry

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "tensorflow"


class TestContractRegistry(unittest.TestCase):
    def test_missing_and_invalid_contract_sources_are_reported(self):
        missing = ContractRegistry("/tmp/definitely-missing-contracts-dir")
        self.assertTrue(missing.load_errors)
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.json"
            p.write_text("{not json", encoding="utf-8")
            reg = ContractRegistry(td)
            self.assertTrue(any("bad.json" in e for e in reg.load_errors))
    def setUp(self):
        self.registry = ContractRegistry(CONTRACTS)
        self.assertGreater(len(self.registry.all_contracts()), 0)

    def test_known_contract(self):
        c = self.registry.get_contract("tf.math.log1p", "independent_api_calls")
        self.assertEqual(c["status"], "SUPPORTED")
        self.assertIn("pure_stateless", c["properties"])
        self.assertEqual(self.registry.get_contract("tf.math.log1p", "shared_readonly_input")["status"], "SUPPORTED")

    def test_unknown_contract(self):
        c = self.registry.get_contract("tf.does.not.exist", "independent_api_calls")
        self.assertEqual(c["status"], "UNKNOWN")
        info = self.registry.validate_oracle_applicability("tf.does.not.exist", "independent_api_calls", "O2")
        self.assertTrue(info["applicable"])
        self.assertEqual(info["contract_status"], "UNKNOWN")

    def test_unsupported_contract(self):
        self.registry.register_contract({
            "api_name": "tf.unsupported.op",
            "scenario": "independent_api_calls",
            "status": "UNSUPPORTED",
            "source": "test",
            "properties": [],
        })
        info = self.registry.validate_oracle_applicability("tf.unsupported.op", "independent_api_calls", "O2")
        self.assertFalse(info["applicable"])
        self.assertEqual(info["contract_status"], "UNSUPPORTED")

    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            self.registry.register_contract({"api_name": "x", "status": "BAD"})

    def test_aggregate(self):
        self.assertEqual(self.registry.aggregate_status("tf.math.log1p", ["independent_api_calls"]), "SUPPORTED")
        self.assertEqual(self.registry.aggregate_status("tf.does.not.exist", ["independent_api_calls"]), "UNKNOWN")

    def test_per_oracle_status_and_plan_gate(self):
        self.registry.register_contract({
            "api_name": "tf.mixed.op", "scenario": "independent_api_calls",
            "status": "SUPPORTED", "source": "test",
            "oracle_status": {"O2": "SUPPORTED", "O3": "UNKNOWN"},
        })
        self.assertEqual(self.registry.get_oracle_status("tf.mixed.op", "independent_api_calls", "O2"), "SUPPORTED")
        self.assertEqual(self.registry.get_oracle_status("tf.mixed.op", "independent_api_calls", "O3"), "UNKNOWN")
        self.assertEqual(self.registry.plan_status("tf.mixed.op", "independent_api_calls", ["O2"]), "SUPPORTED")
        self.assertEqual(self.registry.plan_status("tf.mixed.op", "independent_api_calls", ["O2", "O3"]), "UNKNOWN")

    def test_oracle_allowlist(self):
        self.registry.register_contract({
            "api_name": "tf.allow.op", "scenario": "independent_api_calls",
            "status": "SUPPORTED", "source": "test", "oracles": ["O2"],
        })
        self.assertEqual(self.registry.get_oracle_status("tf.allow.op", "independent_api_calls", "O2"), "SUPPORTED")
        self.assertEqual(self.registry.get_oracle_status("tf.allow.op", "independent_api_calls", "O3"), "UNSUPPORTED")
        self.assertFalse(self.registry.validate_oracle_applicability(
            "tf.allow.op", "independent_api_calls", "O3")["applicable"])


if __name__ == "__main__":
    unittest.main()
