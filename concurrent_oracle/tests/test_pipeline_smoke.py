import os
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.pipeline import ConcurrentOraclePipeline

REAL = Path("/root/fuzz_output_experiment/llm.tf.math.log1p.py")


@unittest.skipUnless(os.environ.get("CO_RUN_TF_TESTS") == "1", "set CO_RUN_TF_TESTS=1")
class TestPipelineSmoke(unittest.TestCase):
    def test_end_to_end_log1p(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = ConcurrentOraclePipeline(output_dir=td)
            summary = pipeline.run_harness(REAL, max_plans=1, run_baseline=True)
            self.assertEqual(summary["decision"], "READY")
            self.assertFalse(summary["errors"], summary["errors"])
            self.assertGreaterEqual(len(summary["results"]), 1)
            r = summary["results"][0]
            self.assertEqual(r["execution_status"], "COMPLETED")
            by = {x["oracle_id"]: x for x in r["oracle_results"]}
            self.assertEqual(by["O2"]["status"], "PASS")
            self.assertEqual(by["O3"]["status"], "PASS")
            self.assertTrue(Path(r["report"]).exists())
            self.assertTrue(Path(r["oracle_result"]).exists())


if __name__ == "__main__":
    unittest.main()
