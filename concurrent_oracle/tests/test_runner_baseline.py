import unittest

from concurrent_oracle.runner import _baseline_bad_cases, baseline_covers_tasks
from concurrent_oracle.schemas import ExecutionObservation, ExecutionStatus, TaskObservation


def obs(status=ExecutionStatus.COMPLETED, baseline=None, tasks=None):
    return ExecutionObservation(plan_id="p", config_id="c", execution_status=status,
                                baseline=baseline or {}, tasks=tasks or [])


class TestRunnerBaseline(unittest.TestCase):
    def test_unused_failed_case_does_not_invalidate(self):
        baseline = {
            "case_000": {"complete": True, "outputs": [1]},
            "case_001": {"complete": False, "outputs": []},
        }
        test = obs(tasks=[TaskObservation(task_id=0, input_id="case_000", status="COMPLETED")])
        self.assertTrue(baseline_covers_tasks(obs(baseline=baseline), test))

    def test_used_failed_case_invalidates(self):
        baseline = {
            "case_000": {"complete": True, "outputs": [1]},
            "case_001": {"complete": False, "outputs": []},
        }
        test = obs(tasks=[TaskObservation(task_id=1, input_id="case_001", status="COMPLETED")])
        self.assertFalse(baseline_covers_tasks(obs(baseline=baseline), test))

    def test_bad_cases_detection(self):
        baseline = {
            "case_000": {"complete": True},
            "case_001": {"complete": False},
            "case_002": {},
        }
        self.assertEqual(_baseline_bad_cases(obs(baseline=baseline)),
                         ["case_001", "case_002"])

    def test_baseline_process_must_complete(self):
        baseline = {"case_000": {"complete": True, "outputs": [1]}}
        test = obs(tasks=[TaskObservation(task_id=0, input_id="case_000", status="COMPLETED")])
        self.assertFalse(baseline_covers_tasks(
            obs(status=ExecutionStatus.TIMEOUT, baseline=baseline), test))


if __name__ == "__main__":
    unittest.main()
