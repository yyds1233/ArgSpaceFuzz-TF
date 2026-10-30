import base64
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from concurrent_oracle.api_suitability_analyzer import APISuitabilityAnalyzer
from concurrent_oracle.contract_registry import ContractRegistry
from concurrent_oracle.harness_analyzer import HarnessAnalyzer
from concurrent_oracle.harness_loader import HarnessLoader
from concurrent_oracle.scenario_generator import ScenarioGenerator
from concurrent_oracle.suitability_rules import SuitabilityRules
from concurrent_oracle.transformer import HarnessTransformer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "llm.tf.math.log1p.py"
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "tensorflow"


def build_generated(tmp: str):
    artifact = HarnessLoader().load(FIXTURE)
    analysis = HarnessAnalyzer().analyze(artifact)
    registry = ContractRegistry(CONTRACTS)
    report = APISuitabilityAnalyzer(SuitabilityRules(), None, registry, {}).analyze(artifact, analysis)
    plan = ScenarioGenerator(contract_registry=registry).generate(report, api_name=artifact.api_name)[0]
    art = HarnessTransformer(tmp).transform(artifact, analysis, plan)
    spec = importlib.util.spec_from_file_location("co_template_types", art.concurrent_harness)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestTemplateSerialization(unittest.TestCase):
    def test_bytes_roundtrip_and_strings(self):
        with tempfile.TemporaryDirectory() as td:
            mod = build_generated(td)
            b64 = base64.b64encode(b"abc").decode()
            self.assertEqual(mod._from_jsonable({"__bytes_b64__": b64}), b"abc")
            b64_bad = base64.b64encode(b"\xff\x00").decode()
            self.assertEqual(mod._from_jsonable({"__bytes_b64__": b64_bad}), b"\xff\x00")
            self.assertEqual(mod._jsonable(np.array(["", "abc"])), ["", "abc"])


@unittest.skipUnless(os.environ.get("CO_RUN_TF_TESTS") == "1", "set CO_RUN_TF_TESTS=1")
class TestTemplateReplayTypes(unittest.TestCase):
    def test_reconstruct_supported_types(self):
        import tensorflow as tf
        with tempfile.TemporaryDirectory() as td:
            mod = build_generated(td)
            self.assertEqual(mod._reconstruct_arg({"kind": "dtype", "name": "float32"}, tf),
                             tf.float32)
            shape = mod._reconstruct_arg({"kind": "tensor_shape", "dims": [None, 2]}, tf)
            self.assertEqual(shape.as_list(), [None, 2])
            self.assertEqual(mod._reconstruct_arg({"kind": "python", "value": 1.0}, tf), 1.0)

            b1 = base64.b64encode(b"").decode()
            b2 = base64.b64encode(b"abc").decode()
            s = mod._reconstruct_arg({
                "kind": "tensor", "dtype": "string", "shape": [2],
                "value": [{"__bytes_b64__": b1}, {"__bytes_b64__": b2}],
            }, tf)
            self.assertEqual(s.numpy().tolist(), [b"", b"abc"])

            sparse = mod._reconstruct_arg({
                "kind": "sparse_tensor", "dtype": "float32", "dense_shape": [2, 2],
                "indices": [[0, 0]], "values": [1.0],
            }, tf)
            self.assertIsInstance(sparse, tf.SparseTensor)

            ragged = mod._reconstruct_arg({
                "kind": "ragged_tensor", "dtype": "float32",
                "flat_values": {"kind": "tensor", "dtype": "float32", "shape": [2],
                                "value": [1.0, 2.0]},
                "nested_row_splits": [[0, 2]],
            }, tf)
            self.assertIsInstance(ragged, tf.RaggedTensor)


if __name__ == "__main__":
    unittest.main()
