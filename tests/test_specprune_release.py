"""CPU-only release manifest consistency checks, not GPU correctness evidence."""

import json
import unittest
from dataclasses import asdict
from pathlib import Path

from cosmos_framework.inference.specprune_exit_compile import compile_exit_layers
from cosmos_framework.inference.specprune_exit_plan import ExitConfig


class ReleaseTest(unittest.TestCase):
    def test_portable_request_reset(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from cosmos_framework.scripts.action_policy_server_specprune import SpecPrunePolicyService

        policy = object.__new__(SpecPrunePolicyService)
        policy.adapter = Mock()
        policy.cfg = SimpleNamespace(seed=0)
        policy.prompt = None
        policy.request_chunk = 0
        policy.enabled = False
        with patch(
            "cosmos_framework.scripts.action_policy_server_specprune.RobolabPolicyService.infer", return_value={}
        ):
            policy.infer({"prompt": "task A"})
            first = policy._rng.integers(100000)
            policy.infer({"prompt": "task A"})
            self.assertEqual(policy.adapter.reset.call_count, 1)
            self.assertEqual(policy.request_chunk, 2)
            policy.infer({"prompt": "task B"})
            self.assertEqual(policy.adapter.reset.call_count, 2)
            self.assertEqual(policy.request_chunk, 1)
            self.assertEqual(policy._rng.integers(100000), first)

    def test_manifest_matches_frozen_runtime(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/specprune_baseline.json").read_text())
        actual = json.loads(json.dumps(asdict(ExitConfig())))
        self.assertEqual(actual, manifest["config"])
        self.assertEqual(len(set(manifest["protocol"]["tasks"])), 10)

    def test_no_padded_compile(self):
        from types import SimpleNamespace

        with self.assertRaises(ValueError):
            compile_exit_layers(SimpleNamespace(net=SimpleNamespace(pad_for_cuda_graphs=True)), cuda_graphs=True)


if __name__ == "__main__":
    unittest.main()
