import sys
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.process_manager import ProcessManager


class TestProcessManager(unittest.TestCase):
    def test_timeout_and_signal(self):
        pm = ProcessManager(sys.executable, preload_path=None)
        r = pm.run([sys.executable, "-c", "import time; time.sleep(5)"],
                   cwd="/tmp", timeout=1)
        self.assertTrue(r.timed_out)
        self.assertEqual(r.signal, "SIGKILL")
        self.assertLess(r.duration, 4.0)

    def test_no_progress_watchdog_and_stack_signal(self):
        with tempfile.TemporaryDirectory() as td:
            events = Path(td) / "events.jsonl"
            code = (
                "import faulthandler, json, os, signal, time\n"
                "faulthandler.register(signal.SIGUSR1, all_threads=True)\n"
                "p = os.environ['EVENTS']\n"
                "with open(p, 'a') as f:\n"
                "    f.write(json.dumps({'event': 'START', 'ts': time.monotonic()}) + '\\n')\n"
                "    f.flush()\n"
                "time.sleep(20)\n"
            )
            pm = ProcessManager(sys.executable, preload_path=None)
            r = pm.run(
                [sys.executable, "-c", code],
                cwd=td,
                timeout=10,
                env_extra={"EVENTS": str(events)},
                events_path=events,
                no_progress_timeout=1,
            )
            self.assertTrue(r.no_progress)
            self.assertTrue(r.timed_out)
            self.assertEqual(r.last_event, "START")
            self.assertLess(r.duration, 6.0)
            self.assertTrue(r.stack_signal_sent)


if __name__ == "__main__":
    unittest.main()
