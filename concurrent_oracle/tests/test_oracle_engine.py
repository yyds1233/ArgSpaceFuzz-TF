import unittest

from concurrent_oracle.oracle_engine import OracleEngine
from concurrent_oracle.schemas import (
    ConcurrentTestPlan,
    ContractStatus,
    ExecutionObservation,
    ExecutionStatus,
    OracleStatus,
    TaskObservation,
)


def desc(value):
    return {"kind": "tensor", "dtype": "float32", "shape": [len(value)],
            "sha256": f"h-{value!r}", "value": value}


def task(tid, input_id, value, status="COMPLETED"):
    return TaskObservation(
        task_id=tid, input_id=input_id, status=status,
        metadata={"outputs": [{"call": "tf.x", "output": desc(value)}],
                  "complete": status == "COMPLETED",
                  "trials_expected": 1,
                  "trials_completed": 1 if status == "COMPLETED" else 0},
    )


def plan(oracles=("O2", "O3"), threads=2, contract=ContractStatus.SUPPORTED,
         oracle_contracts=None):
    return ConcurrentTestPlan(
        plan_id="p1", artifact_id="a1", scenario="independent_api_calls",
        runtime={"external_threads": threads, "intra_op_threads": 1, "inter_op_threads": 1,
                 "tf_data_parallel_calls": 1, "config_id": "C03"},
        oracles=list(oracles), contract_status=contract,
        oracle_contracts=oracle_contracts or {},
    )


def obs(tasks, baseline=None, complete=True, status=ExecutionStatus.COMPLETED):
    if baseline is None:
        baseline = {
            "case_000": {"outputs": [{"call": "tf.x", "output": desc([1.0])}]},
            "case_001": {"outputs": [{"call": "tf.x", "output": desc([2.0])}]},
        }
    return ExecutionObservation(
        plan_id="p1", config_id="C03", execution_status=status,
        observation_complete=complete, tasks=tasks,
        task_progress={"submitted": len(tasks), "started": len(tasks),
                       "completed": len(tasks), "failed": 0, "pending": 0},
        baseline=baseline,
    )


class TestOracleEngine(unittest.TestCase):
    def setUp(self):
        self.engine = OracleEngine()

    def test_pass(self):
        r = self.engine.evaluate(obs([task(0, "case_000", [1.0]), task(1, "case_001", [2.0])]), plan())
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O2"].status, OracleStatus.PASS)
        self.assertEqual(by["O3"].status, OracleStatus.PASS)

    def test_output_mismatch_violation(self):
        r = self.engine.evaluate(obs([task(0, "case_000", [1.0]), task(1, "case_001", [99.0])]), plan())
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O2"].status, OracleStatus.VIOLATION)
        self.assertEqual(by["O3"].status, OracleStatus.VIOLATION)

    def test_cross_talk_violation(self):
        r = self.engine.evaluate(obs([task(0, "case_000", [1.0]), task(1, "case_001", [1.0])]), plan())
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O3"].status, OracleStatus.VIOLATION)
        self.assertIn("suspected_cross_talk", by["O3"].evidence)

    def test_non_injective_output_is_not_cross_talk(self):
        baseline = {
            "case_000": {"outputs": [{"call": "tf.x", "output": desc([0.0])}]},
            "case_001": {"outputs": [{"call": "tf.x", "output": desc([0.0])}]},
        }
        r = self.engine.evaluate(
            obs([task(0, "case_000", [0.0]), task(1, "case_001", [0.0])], baseline=baseline),
            plan(),
        )
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O3"].status, OracleStatus.PASS)

    def test_unknown_contract_suspected_cross_talk_inconclusive(self):
        r = self.engine.evaluate(
            obs([task(0, "case_000", [1.0]), task(1, "case_001", [1.0])]),
            plan(contract=ContractStatus.UNKNOWN),
        )
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O3"].status, OracleStatus.INCONCLUSIVE)
        self.assertIn("suspected_cross_talk", by["O3"].evidence)

    def test_incomplete_inconclusive(self):
        r = self.engine.evaluate(obs([task(0, "case_000", [1.0])], complete=False), plan())
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O2"].status, OracleStatus.INCONCLUSIVE)
        self.assertEqual(by["O3"].status, OracleStatus.INCONCLUSIVE)

    def test_not_applicable(self):
        r = self.engine.evaluate(obs([task(0, "case_000", [1.0]), task(1, "case_001", [2.0])]),
                                 plan(oracles=("O3",)))
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O2"].status, OracleStatus.NOT_APPLICABLE)

    def test_per_oracle_contract_gate(self):
        r = self.engine.evaluate(
            obs([task(0, "case_000", [1.0]), task(1, "case_001", [99.0])]),
            plan(contract=ContractStatus.UNKNOWN,
                 oracle_contracts={"O2": "SUPPORTED", "O3": "UNKNOWN"}),
        )
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O2"].status, OracleStatus.VIOLATION)
        self.assertEqual(by["O3"].status, OracleStatus.INCONCLUSIVE)

    def test_partial_trial_is_inconclusive(self):
        r = self.engine.evaluate(
            obs([task(0, "case_000", [1.0]),
                 task(1, "case_001", [2.0], status="PARTIAL")]),
            plan(),
        )
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O2"].status, OracleStatus.INCONCLUSIVE)
        self.assertEqual(by["O3"].status, OracleStatus.INCONCLUSIVE)

    def test_structural_problem_is_inconclusive(self):
        r = self.engine.evaluate(
            obs([task(0, "case_000", [1.0])]), plan(threads=2),
        )
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O3"].status, OracleStatus.INCONCLUSIVE)
        self.assertIn("observation_integrity", by["O3"].evidence)

    def test_missing_oracle_explicit(self):
        r = self.engine.evaluate(obs([task(0, "case_000", [1.0]), task(1, "case_001", [2.0])]),
                                 plan(oracles=("O2", "O3", "O4")))
        by = {e.oracle_id: e for e in r.oracle_results}
        self.assertEqual(by["O4"].status, OracleStatus.INCONCLUSIVE)
        self.assertIn("not implemented", by["O4"].reason)


if __name__ == "__main__":
    unittest.main()
