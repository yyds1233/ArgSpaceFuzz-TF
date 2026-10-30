import json
import shutil
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.harness_loader import HarnessLoader

REAL = Path("/root/fuzz_output_experiment/llm.tf.math.log1p.py")


class TestHarnessLoader(unittest.TestCase):
    def setUp(self):
        self.loader = HarnessLoader()

    def test_load_real_harness(self):
        self.assertTrue(REAL.exists(), REAL)
        a = self.loader.load(REAL)
        self.assertEqual(a.framework, "tensorflow")
        self.assertEqual(a.api_name, "tf.math.log1p")
        self.assertTrue(a.harness_hash.startswith("sha256:"))
        self.assertTrue(a.artifact_id.startswith("tf.math.log1p_"))
        self.assertEqual(a.source, "fuzz_api_one")
        self.assertTrue(Path(a.harness_path).is_absolute())
        self.assertIsNotNone(a.psr_path)
        self.assertTrue(Path(a.psr_path).exists())

    def test_load_directory_with_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            h1 = td / "llm.tf.nn.relu.py"
            h2 = td / "llm.tf.math.log1p.py"
            shutil.copy2("/root/fuzz_output_experiment/llm.tf.nn.relu.py", h1)
            shutil.copy2(REAL, h2)
            manifest = td / "manifest.json"
            manifest.write_text(json.dumps([
                {"api": "tf.nn.relu", "harness_path": str(h1)},
                {"api_name": "tf.math.log1p", "harness_path": str(h2)},
            ]), encoding="utf-8")
            arts = self.loader.load_directory(td, manifest)
            self.assertEqual(len(arts), 2)
            self.assertEqual({a.api_name for a in arts}, {"tf.nn.relu", "tf.math.log1p"})

    def test_scan_directory(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            shutil.copy2(REAL, td / "llm.tf.math.log1p.py")
            arts = self.loader.load_directory(td)
            self.assertEqual(len(arts), 1)
            self.assertEqual(arts[0].api_name, "tf.math.log1p")

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.loader.load("/tmp/does-not-exist-harness.py")


if __name__ == "__main__":
    unittest.main()
