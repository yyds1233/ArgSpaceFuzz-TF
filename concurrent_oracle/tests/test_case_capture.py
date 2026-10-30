import json
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.case_capture import (
    CaseCapture,
    check_case_eligibility,
    count_unsupported_target_cases,
)


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

    def test_case_eligibility(self):
        good = {
            "target_calls": [{
                "call": "tf.math.log1p",
                "args": [{"kind": "tensor", "dtype": "float32", "shape": [2],
                          "value": [1.0, 2.0], "sha256": "abc"}],
                "kwargs": {},
            }]
        }
        self.assertTrue(check_case_eligibility(good)["eligible"])
        truncated = {
            "target_calls": [{
                "call": "tf.math.log1p",
                "args": [{"kind": "tensor", "dtype": "float32", "shape": [2],
                          "value": None, "value_truncated": True, "sha256": "abc"}],
                "kwargs": {},
            }]
        }
        self.assertFalse(check_case_eligibility(truncated)["eligible"])
        unsupported = {
            "target_calls": [{
                "call": "tf.x",
                "args": [{"kind": "unsupported_input", "type": "Variable"}],
                "kwargs": {},
            }]
        }
        self.assertFalse(check_case_eligibility(unsupported)["eligible"])
        ndarray = {
            "target_calls": [{
                "call": "tf.x",
                "args": [{"kind": "ndarray", "dtype": "float32", "shape": [1],
                          "value": [1.0], "sha256": "abc"}],
                "kwargs": {},
            }]
        }
        r = check_case_eligibility(ndarray)
        self.assertTrue(r["eligible"])
        self.assertEqual(r["conversions"], ["ndarray_to_tensor"])
        self.assertFalse(check_case_eligibility({"target_calls": []})["eligible"])


if __name__ == "__main__":
    unittest.main()
