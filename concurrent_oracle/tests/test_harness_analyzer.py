import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader

REAL = Path("/root/fuzz_output_experiment/llm.tf.math.log1p.py")


class TestHarnessAnalyzer(unittest.TestCase):
    def setUp(self):
        self.loader = HarnessLoader()
        self.analyzer = HarnessAnalyzer()

    def test_analyze_real_harness(self):
        artifact = self.loader.load(REAL)
        a = self.analyzer.analyze(artifact)
        self.assertEqual(a.ast_parse_status, "SUCCESS")
        self.assertTrue(a.target_calls)
        names = {c.api_name for c in a.target_calls}
        self.assertIn("tf.math.log1p", names)
        self.assertTrue(all(c.is_target for c in a.target_calls))
        self.assertTrue(a.input_variables)
        self.assertTrue(a.tensor_construction)
        self.assertTrue(a.output_expressions or a.cleanup_expressions)
        self.assertFalse(any("no target call" in w for w in a.warnings))

    def test_syntax_error(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "llm.tf.bad.py"
            p.write_text("def broken(:\n    pass\n", encoding="utf-8")
            artifact = self.loader.load(p, api_name="tf.bad")
            a = self.analyzer.analyze(artifact)
            self.assertEqual(a.ast_parse_status, "FAILED")
            self.assertTrue(any("SyntaxError" in w for w in a.warnings))

    def test_no_target_call_warning(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "llm.tf.nothing.py"
            p.write_text("import tensorflow as tf\nx = 1\n", encoding="utf-8")
            artifact = self.loader.load(p, api_name="tf.nothing")
            a = self.analyzer.analyze(artifact)
            self.assertTrue(any("no target call" in w for w in a.warnings))


if __name__ == "__main__":
    unittest.main()
