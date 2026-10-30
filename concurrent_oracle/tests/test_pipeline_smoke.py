import os
import tempfile
import unittest
from pathlib import Path

from concurrent_oracle.pipeline import ConcurrentOraclePipeline

REAL = Path(__file__).resolve().parent / "fixtures" / "llm.tf.math.log1p.py"
UNKNOWN_CONTRACT = Path(__file__).resolve().parent / "fixtures" / "llm.tf.math.exp.py"


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

            # R03/R04/R05: target args must be captured; baseline and test are
            # separate processes; trials must reach the execution loop.
            cap = summary["captures"][0]
            self.assertGreaterEqual(cap["cases_with_target_calls"], 1)
            self.assertGreaterEqual(cap["cases_eligible"], 1)
            self.assertEqual(cap["cases_ineligible"], 0)
            self.assertNotEqual(r["baseline_run_id"], r["test_run_id"])
            self.assertEqual(r["trials"], 10)
            plan_id = r["plan_id"]
            base_dirs = sorted((Path(td) / "executions" / plan_id / "baseline").glob("*"))
            test_dirs = sorted((Path(td) / "executions" / plan_id / "concurrent").glob("*"))
            self.assertTrue(base_dirs and test_dirs)
            import json
            b = json.loads((base_dirs[-1] / "execution.json").read_text())
            t_obs = json.loads((test_dirs[-1] / "execution.json").read_text())
            self.assertEqual(b["mode"], "baseline")
            self.assertEqual(t_obs["mode"], "concurrent")
            self.assertNotEqual(b["runtime"]["pid"], t_obs["runtime"]["pid"])
            self.assertEqual(b["trials"], 10)
            self.assertEqual(t_obs["trials"], 10)
            self.assertEqual(t_obs["runtime"]["deterministic_requested"], True)
            self.assertEqual(t_obs["runtime"]["deterministic_applied"], True)
            combined = json.loads((test_dirs[-1] / "combined_observation.json").read_text())
            self.assertTrue(combined["baseline"])
            self.assertEqual(combined["baseline_run_id"], b["run_id"])
            self.assertEqual(combined["test_run_id"], t_obs["run_id"])
            event_names = []
            events_path = test_dirs[-1] / "events.jsonl"
            if events_path.exists():
                with open(events_path, encoding="utf-8") as f:
                    event_names = [json.loads(line)["event"] for line in f if line.strip()]
            for required in ("ARG_REBUILD_START", "ARG_REBUILD_END",
                             "TARGET_ENTER", "TARGET_RETURN",
                             "DESCRIBE_START", "DESCRIBE_END",
                             "BARRIER_WAIT", "BARRIER_RELEASED",
                             "TASK_FINISHED"):
                self.assertIn(required, event_names)

    def test_explore_mode_runs_unknown_contract(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = ConcurrentOraclePipeline(output_dir=td)
            summary = pipeline.run_harness(UNKNOWN_CONTRACT, max_plans=1,
                                           allow_unknown_contract=True)
            self.assertEqual(summary["decision"], "CONDITIONAL")
            self.assertTrue(summary["captures"], summary)
            self.assertGreaterEqual(len(summary["results"]), 1, summary)
            self.assertGreaterEqual(summary["captures"][0]["cases_with_target_calls"], 1)

    def test_shared_readonly_backend(self):
        with tempfile.TemporaryDirectory() as td:
            pipeline = ConcurrentOraclePipeline(output_dir=td)
            artifact, analysis, report, plans = pipeline.analyze(REAL)
            plan = next(p for p in plans if p.scenario == "shared_readonly_input")
            art = pipeline.generate(artifact, analysis, plan)
            cap = pipeline.runner.capture_cases(art, plan)
            self.assertGreaterEqual(cap["cases_with_target_calls"], 1)
            obs = pipeline.run(art, plan)
            result = pipeline.evaluate(obs, plan)
            by = {x.oracle_id: x for x in result.oracle_results}
            self.assertEqual(obs.execution_status.value, "COMPLETED")
            self.assertEqual(by["O3"].status.value, "PASS")
            self.assertIsNotNone(obs.diagnostics.get("shared_input_id"))
            input_ids = {t.input_id for t in obs.tasks}
            self.assertEqual(len(input_ids), 1)


if __name__ == "__main__":
    unittest.main()
