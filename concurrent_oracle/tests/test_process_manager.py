import sys
import unittest

from concurrent_oracle.process_manager import ProcessManager


class TestProcessManager(unittest.TestCase):
    def test_timeout_and_signal(self):
        pm = ProcessManager(sys.executable, preload_path=None)
        r = pm.run([sys.executable, "-c", "import time; time.sleep(5)"],
                   cwd="/tmp", timeout=1)
        self.assertTrue(r.timed_out)
        self.assertEqual(r.signal, "SIGKILL")
        self.assertLess(r.duration, 4.0)


if __name__ == "__main__":
    unittest.main()
