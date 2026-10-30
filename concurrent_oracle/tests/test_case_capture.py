import json
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.case_capture import CaseCapture, count_unsupported_target_cases


class TestCaseCapture(unittest.TestCase):
    def test_merge_captured_target_calls(self):
        with tempfile.TemporaryDirectory() as td:
            cap = CaseCapture(td)
            p = cap.write_cases("plan1")
            captured = {
                "case_000": {
                    "target_calls": [{"call": "tf.math.log1p", "args": [], "kwargs": {}}],
                    "source": "captured_target_call",
                },
                "case_001": {"target_calls": [], "source": "no_target_call"},
            }
            cap.merge_captured_cases("plan1", captured, {"capture_run_id": "r1"})
            data = json.loads(Path(p).read_text())
            by = {c["case_id"]: c for c in data["cases"]}
            self.assertEqual(by["case_000"]["case_source"], "captured_target_call")
            self.assertEqual(len(by["case_000"]["target_calls"]), 1)
            self.assertEqual(by["case_001"]["case_source"], "no_target_call")
            self.assertEqual(data["capture_metadata"]["capture_run_id"], "r1")
            self.assertEqual(count_unsupported_target_cases(captured), 0)
            unsupported = {
                "case_002": {
                    "target_calls": [{"call": "tf.x",
                                      "args": [{"kind": "unsupported_input", "type": "Variable"}],
                                      "kwargs": {}}],
                    "source": "captured_target_call",
                }
            }
            self.assertEqual(count_unsupported_target_cases(unsupported), 1)


if __name__ == "__main__":
    unittest.main()
