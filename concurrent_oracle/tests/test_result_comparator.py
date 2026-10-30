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

    def test_non_injective_output_is_not_cross_talk(self):
        baseline = {
            "case_000": {"outputs": [{"call": "tf.x", "output": desc([0.0])}]},
            "case_001": {"outputs": [{"call": "tf.x", "output": desc([0.0])}]},
        }
        task = TaskObservation(
            task_id=1, input_id="case_001", status="COMPLETED",
            metadata={"outputs": [{"call": "tf.x", "output": desc([0.0])}]},
        )
        self.assertEqual(
            find_cross_talk(task, baseline, own_baseline=baseline["case_001"],
                            candidate_input_ids=["case_000", "case_001"]),
            [],
        )

    def test_describe_error_is_inconclusive(self):
        r = compare_described(
            {"kind": "describe_error", "error_type": "TypeError", "error": "x"},
            {"kind": "describe_error", "error_type": "TypeError", "error": "x"},
        )
        self.assertFalse(r["equal"])
        self.assertTrue(r["inconclusive"])

    def test_sparse_and_ragged_descriptions(self):
        sparse = {
            "kind": "sparse_tensor", "dtype": "float32", "dense_shape": [2, 2],
            "indices": [[0, 0], [1, 1]], "values": [1.0, 2.0],
        }
        self.assertTrue(compare_described(sparse, sparse)["equal"])
        bad = dict(sparse, values=[1.0, 3.0])
        self.assertFalse(compare_described(sparse, bad)["equal"])
        ragged = {
            "kind": "ragged_tensor", "dtype": "float32",
            "flat_values": desc([1.0, 2.0, 3.0]),
            "row_splits": [0, 2, 3],
            "nested_row_splits": [[0, 2, 3]],
        }
        self.assertTrue(compare_described(ragged, ragged)["equal"])
        bad_ragged = dict(ragged, nested_row_splits=[[0, 1, 3]])
        self.assertFalse(compare_described(ragged, bad_ragged)["equal"])

    def test_complex_and_large_integer(self):
        c1 = desc([{"real": 1.0, "imag": 2.0}], dtype="complex64")
        c2 = desc([{"real": 1.0 + 1e-7, "imag": 2.0}], dtype="complex64")
        self.assertTrue(compare_described(c1, c2)["equal"])
        big = desc(list(range(5000)), dtype="int32")
        self.assertTrue(compare_described(big, big)["equal"])
        self.assertFalse(compare_described(big, desc(list(range(5000)), dtype="int64"))["equal"])

    def test_describe_error_is_inconclusive(self):
        expected = {"kind": "list", "items": [desc([1.0]), desc([2.0])]}
        actual = {"kind": "list", "items": [desc([1.0 + 1e-7]), desc([2.0])]}
        r = compare_described(expected, actual)
        self.assertTrue(r["equal"], r)
        actual2 = {"kind": "list", "items": [desc([1.1]), desc([2.0])]}
        r2 = compare_described(expected, actual2)
        self.assertFalse(r2["equal"], r2)


if __name__ == "__main__":
    unittest.main()
