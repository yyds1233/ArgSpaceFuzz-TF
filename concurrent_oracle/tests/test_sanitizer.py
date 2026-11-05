import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.runner import ConcurrentRunner


class TestSanitizerParser(unittest.TestCase):
    def test_disabled(self):
        with tempfile.TemporaryDirectory() as td:
            r = ConcurrentRunner(td, preload_path=None, sanitizer={"enabled": False})
            rep = r._sanitizer_report("ERROR: AddressSanitizer: x", 1, None, Path("e.log"))
            self.assertFalse(rep["enabled"])
            self.assertFalse(rep["detected"])

    def test_asan_report(self):
        with tempfile.TemporaryDirectory() as td:
            r = ConcurrentRunner(
                td, preload_path=None,
                sanitizer={"enabled": True, "type": "asan", "build_id": "b1"},
            )
            text = (
                "ERROR: AddressSanitizer: heap-buffer-overflow on address 0x1\n"
                "    #0 0x1 in foo\n"
                "    #1 0x2 in bar\n"
                "SUMMARY: AddressSanitizer: heap-buffer-overflow\n"
            )
            rep = r._sanitizer_report(text, 1, None, Path("e.log"))
            self.assertTrue(rep["detected"])
            self.assertEqual(rep["report_type"], "AddressSanitizer")
            self.assertIn("heap-buffer-overflow", rep["summary"])
            self.assertEqual(len(rep["stack"]), 2)
            self.assertTrue(rep["instrumented"])

    def test_tsan_report(self):
        with tempfile.TemporaryDirectory() as td:
            r = ConcurrentRunner(
                td, preload_path=None,
                sanitizer={"enabled": True, "type": "tsan", "build_id": "b2"},
            )
            rep = r._sanitizer_report(
                "WARNING: ThreadSanitizer: data race\n    #0 0x1 in foo\n", 66, "SIGABRT",
                Path("e.log"),
            )
            self.assertTrue(rep["detected"])
            self.assertEqual(rep["report_type"], "ThreadSanitizer")
            self.assertEqual(rep["signal"], "SIGABRT")


if __name__ == "__main__":
    unittest.main()
