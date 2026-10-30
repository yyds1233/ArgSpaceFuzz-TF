import unittest

from concurrent_oracle.result_comparator import (
    compare_described,
    compare_output_lists,
    find_cross_talk,
)
from concurrent_oracle.schemas import TaskObservation


def desc(value, dtype="float32", shape=None, sha=None):
    if shape is None:
        shape = [len(value)] if isinstance(value, list) else []
    return {
        "kind": "tensor",
        "dtype": dtype,
        "shape": shape,
        "sha256": sha or f"hash-{value!r}",
        "value": value,
    }


class TestResultComparator(unittest.TestCase):
    def test_equal_values_with_different_hash(self):
        r = compare_described(desc([1.0, 2.0], sha="a"), desc([1.0, 2.0], sha="b"))
        self.assertTrue(r["equal"], r)

    def test_float_tolerance(self):
        r = compare_described(desc([1.0]), desc([1.0 + 1e-7]))
        self.assertTrue(r["equal"], r)
        r = compare_described(desc([1.0]), desc([1.1]))
        self.assertFalse(r["equal"], r)

    def test_nan_and_inf(self):
        self.assertTrue(compare_described(desc([float("nan")]), desc([float("nan")]))["equal"])
        self.assertFalse(compare_described(desc([float("nan")]), desc([1.0]))["equal"])
        self.assertTrue(compare_described(desc([float("inf")]), desc([float("inf")]))["equal"])
        self.assertFalse(compare_described(desc([float("inf")]), desc([float("-inf")]))["equal"])

    def test_shape_dtype_mismatch(self):
        self.assertFalse(compare_described(desc([1.0], shape=[1]), desc([1.0], shape=[2]))["equal"])
        self.assertFalse(compare_described(desc([1.0], dtype="float32"),
                                           desc([1.0], dtype="float64"))["equal"])

    def test_output_call_count(self):
        e = [{"call": "tf.math.log1p", "output": desc([1.0])}]
        a = [{"call": "tf.math.log1p", "output": desc([1.0])},
             {"call": "tf.raw_ops.Log1p", "output": desc([1.0])}]
        self.assertFalse(compare_output_lists(e, a)["equal"])

    def test_cross_talk(self):
        baseline = {
            "case_000": {"outputs": [{"call": "tf.x", "output": desc([1.0])}]},
            "case_001": {"outputs": [{"call": "tf.x", "output": desc([2.0])}]},
        }
        task = TaskObservation(
            task_id=1, input_id="case_001", status="COMPLETED",
            metadata={"outputs": [{"call": "tf.x", "output": desc([1.0])}]},
        )
        self.assertEqual(find_cross_talk(task, baseline), ["case_000"])


if __name__ == "__main__":
    unittest.main()
