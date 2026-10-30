import unittest

from concurrent_oracle.adapters.tensorflow import TensorFlowAdapter


class TestTensorFlowAdapter(unittest.TestCase):
    def test_raw_value_comparison_matches_oracle_rules(self):
        a = TensorFlowAdapter()
        self.assertTrue(a.compare_outputs([1.0, 2.0], [1.0 + 1e-7, 2.0])["equal"])
        self.assertFalse(a.compare_outputs([1.0], [1.1])["equal"])
        self.assertTrue(a.compare_outputs([float("nan")], [float("nan")])["equal"])
        self.assertFalse(a.compare_outputs([float("nan")], [1.0])["equal"])
        self.assertTrue(a.compare_outputs([float("inf")], [float("inf")])["equal"])
        self.assertFalse(a.compare_outputs([float("inf")], [float("-inf")])["equal"])
        self.assertFalse(a.compare_outputs({"a": [1]}, {"a": [2]})["equal"])


if __name__ == "__main__":
    unittest.main()
