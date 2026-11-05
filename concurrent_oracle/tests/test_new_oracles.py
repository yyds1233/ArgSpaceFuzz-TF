import unittest

from concurrent_oracle.oracles.callback_lifecycle import CallbackLifecycleOracle
from concurrent_oracle.oracles.exception_recovery import ExceptionRecoveryOracle
from concurrent_oracle.oracles.input_preservation import InputPreservationOracle
from concurrent_oracle.oracles.invalid_input import InvalidInputOracle
from concurrent_oracle.oracles.memory_safety import MemorySafetyOracle
from concurrent_oracle.oracles.tfdata_integrity import TFDataIntegrityOracle
from concurrent_oracle.schemas import (
    ConcurrentTestPlan,
    ExecutionObservation,
    ExecutionStatus,
    OracleStatus,
    TaskObservation,
)


def plan(oracle, contract="SUPPORTED", scenario="unit", metadata=None):
    return ConcurrentTestPlan(
        plan_id="p",
        artifact_id="a",
        scenario=scenario,
        oracles=[oracle],
        contract_status=contract,
        oracle_contracts={oracle: contract},
        runtime={"external_threads": 2},
        metadata=metadata or {},
    )


def obs(tasks, status=ExecutionStatus.COMPLETED, complete=True, diagnostics=None):
    return ExecutionObservation(
        plan_id="p",
        config_id="c",
        execution_status=status,
        observation_complete=complete,
        tasks=tasks,
        diagnostics=diagnostics or {},
    )


def task(tid, status="COMPLETED", metadata=None):
    return TaskObservation(task_id=tid, input_id=f"case_{tid}", status=status,
                           metadata=metadata or {})


class TestO4InputPreservation(unittest.TestCase):
    def test_pass_and_violation(self):
        o = InputPreservationOracle()
        p = plan("O4")
        t = task(0, metadata={"input_preservation": {"checked": 1, "preserved": True}})
        r = o.evaluate(p, obs([t]))
        self.assertEqual(r.status, OracleStatus.PASS)

        t = task(0, metadata={"input_preservation": {
            "checked": 1, "preserved": False,
            "diffs": [{"arg": "args[0]", "before": "a", "after": "b"}],
        }})
        r = o.evaluate(p, obs([t]))
        self.assertEqual(r.status, OracleStatus.VIOLATION)

    def test_unknown_contract_and_missing(self):
        o = InputPreservationOracle()
        p = plan("O4", contract="UNKNOWN")
        t = task(0, metadata={"input_preservation": {"preserved": False, "diffs": []}})
        self.assertEqual(o.evaluate(p, obs([t])).status, OracleStatus.INCONCLUSIVE)
        self.assertEqual(o.evaluate(p, obs([task(0)])).status, OracleStatus.INCONCLUSIVE)


class TestO1InvalidInput(unittest.TestCase):
    def test_pass_and_accept(self):
        o = InvalidInputOracle()
        p = plan("O1")
        t = task(0, status="EXCEPTION", metadata={"invalid_input": {
            "observed_exception": "InvalidArgumentError", "accepted": False,
        }})
        self.assertEqual(o.evaluate(p, obs([t])).status, OracleStatus.PASS)

        t = task(0, metadata={"invalid_input": {"accepted": True}})
        self.assertEqual(o.evaluate(p, obs([t])).status, OracleStatus.VIOLATION)

    def test_native_crash_is_inconclusive(self):
        o = InvalidInputOracle()
        p = plan("O1")
        r = o.evaluate(p, obs([], status=ExecutionStatus.CRASH))
        self.assertEqual(r.status, OracleStatus.INCONCLUSIVE)


class TestO5ExceptionRecovery(unittest.TestCase):
    def _tasks(self, c_status="COMPLETED"):
        return [
            task(0, metadata={"phase": "A"}),
            task(1, status="EXCEPTION", metadata={
                "phase": "B",
                "invalid_input": {"observed_exception": "InvalidArgumentError"},
            }),
            task(2, status=c_status, metadata={"phase": "C"}),
        ]

    def test_pass_and_recovery_failure(self):
        o = ExceptionRecoveryOracle()
        p = plan("O5")
        self.assertEqual(o.evaluate(p, obs(self._tasks())).status, OracleStatus.PASS)
        self.assertEqual(o.evaluate(p, obs(self._tasks("EXCEPTION"))).status,
                         OracleStatus.VIOLATION)

    def test_missing_phase(self):
        o = ExceptionRecoveryOracle()
        p = plan("O5")
        self.assertEqual(o.evaluate(p, obs([task(0, metadata={"phase": "A"})])).status,
                         OracleStatus.INCONCLUSIVE)


class TestO6TFDataIntegrity(unittest.TestCase):
    def _ds(self, **kw):
        base = {
            "kind": "parallel_map",
            "expected_cardinality": 4,
            "observed_cardinality": 4,
            "expected_ids": [0, 1, 2, 3],
            "observed_ids": [0, 1, 2, 3],
            "deterministic": True,
            "order_preserved": True,
        }
        base.update(kw)
        return task(0, metadata={"dataset": base})

    def test_pass_and_problems(self):
        o = TFDataIntegrityOracle()
        p = plan("O6")
        self.assertEqual(o.evaluate(p, obs([self._ds()])).status, OracleStatus.PASS)
        self.assertEqual(o.evaluate(p, obs([self._ds(observed_ids=[0, 1, 2])])).status,
                         OracleStatus.VIOLATION)
        self.assertEqual(o.evaluate(p, obs([self._ds(order_preserved=False)])).status,
                         OracleStatus.VIOLATION)

    def test_unknown_contract(self):
        o = TFDataIntegrityOracle()
        p = plan("O6", contract="UNKNOWN")
        r = o.evaluate(p, obs([self._ds(observed_ids=[0, 1])]))
        self.assertEqual(r.status, OracleStatus.INCONCLUSIVE)


class TestO7CallbackLifecycle(unittest.TestCase):
    def test_pass_and_mismatch(self):
        o = CallbackLifecycleOracle()
        p = plan("O7")
        good = task(0, metadata={"callbacks": {
            "entered_ids": [0, 1], "exited_ids": [0],
            "expected_exception_at": 1, "exception_propagated": True,
            "subsequent_pipeline_ok": True,
        }})
        self.assertEqual(o.evaluate(p, obs([good])).status, OracleStatus.PASS)
        bad = task(0, metadata={"callbacks": {
            "entered_ids": [0, 1], "exited_ids": [0],
            "subsequent_pipeline_ok": False,
        }})
        self.assertEqual(o.evaluate(p, obs([bad])).status, OracleStatus.VIOLATION)


class TestO9MemorySafety(unittest.TestCase):
    def test_disabled_detected_pass(self):
        o = MemorySafetyOracle()
        p = plan("O9")
        self.assertEqual(o.evaluate(p, obs([], diagnostics={})).status,
                         OracleStatus.INCONCLUSIVE)
        detected = obs([], diagnostics={"sanitizer": {
            "enabled": True, "detected": True, "report_type": "AddressSanitizer",
            "summary": "heap-buffer-overflow",
        }})
        self.assertEqual(o.evaluate(p, detected).status, OracleStatus.VIOLATION)
        clean = obs([], diagnostics={"sanitizer": {"enabled": True, "detected": False}})
        self.assertEqual(o.evaluate(p, clean).status, OracleStatus.PASS)


if __name__ == "__main__":
    unittest.main()
