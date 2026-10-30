import json
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle import constants
from concurrent_oracle.schemas import (
    ConcurrentTestPlan,
    Decision,
    ExecutionObservation,
    ExecutionStatus,
    HarnessArtifact,
    OracleEntry,
    OracleResult,
    OracleStatus,
    SuitabilityReport,
    TaskObservation,
    make_artifact_id,
    to_jsonable,
    validate_oracles,
    validate_state_model,
    write_json,
)


class TestSchemas(unittest.TestCase):
    def test_artifact_roundtrip(self):
        a = HarnessArtifact(
            artifact_id="tf_test_12345678",
            api_name="tf.math.log1p",
            harness_path="/tmp/h.py",
            harness_hash="sha256:abc",
        )
        d = to_jsonable(a)
        self.assertEqual(d["framework"], "tensorflow")
        self.assertEqual(HarnessArtifact.from_dict(d).api_name, "tf.math.log1p")

    def test_report_plan_result_json(self):
        report = SuitabilityReport(
            artifact_id="a1",
            decision=Decision.READY,
            rule_candidates=["O2"],
            candidate_scenarios=[],
        )
        plan = ConcurrentTestPlan(plan_id="p1", artifact_id="a1", scenario="independent_api_calls",
                                  oracles=["O2"], trials=3)
        obs = ExecutionObservation(
            plan_id="p1", config_id="c1", execution_status=ExecutionStatus.COMPLETED,
            tasks=[TaskObservation(task_id=0, input_id="case_0", status="COMPLETED")],
        )
        result = OracleResult(
            plan_id="p1", scenario="independent_api_calls", execution_status=ExecutionStatus.COMPLETED,
            oracle_results=[OracleEntry(oracle_id="O2", status=OracleStatus.PASS, reason="ok")],
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"
            write_json(p, {"report": report, "plan": plan, "obs": obs, "result": result})
            data = json.loads(p.read_text())
            self.assertEqual(data["report"]["decision"], "READY")
            self.assertEqual(data["plan"]["scenario"], "independent_api_calls")
            self.assertEqual(data["obs"]["execution_status"], "COMPLETED")
            self.assertEqual(data["result"]["oracle_results"][0]["status"], "PASS")
            self.assertEqual(SuitabilityReport.from_dict(data["report"]).decision, Decision.READY)
            self.assertEqual(ConcurrentTestPlan.from_dict(data["plan"]).oracles, ["O2"])
            self.assertEqual(ExecutionObservation.from_dict(data["obs"]).tasks[0].task_id, 0)
            self.assertEqual(OracleResult.from_dict(data["result"]).oracle_results[0].status, OracleStatus.PASS)

    def test_validation(self):
        self.assertEqual(validate_oracles(["O2", "O3"]), ["O2", "O3"])
        with self.assertRaises(ValueError):
            validate_oracles(["O99"])
        self.assertEqual(validate_state_model("INDEPENDENT_INPUTS"), "INDEPENDENT_INPUTS")
        with self.assertRaises(ValueError):
            validate_state_model("BAD")
        self.assertTrue(make_artifact_id("tf.math.log1p", "sha256:abcdef123").startswith("tf.math.log1p_"))
        self.assertIn("O1", constants.ORACLE_IDS)


if __name__ == "__main__":
    unittest.main()
