"""Exercise real client/server protocol wrappers with a fake transport on CPU."""

import ast
import sys
import threading
import unittest
from dataclasses import make_dataclass
from types import ModuleType, SimpleNamespace

import numpy as np
from test_c3ache import BASELINE, ROOT, C3acheCache, C3acheConfig, load_file

base = load_file("robolab_base_under_test", BASELINE.parent / "RoboLab/robolab/eval/base_client.py")


class BaselineClient(base.InferenceClient):
    """Use the real RoboLab chunk/reset flow with no websocket or simulator."""

    open_loop_horizon = 2

    def __init__(self, **kwargs):
        """Record transport requests and optionally simulate a lost reset message."""
        super().__init__()
        self.requests = []
        self.fail_reset = False

    def _extract_observation(self, raw_obs, *, env_id=0):
        """Pass a tiny observation through instead of camera transforms."""
        return dict(raw_obs)

    def _pack_request(self, extracted_obs, instruction):
        """Retain observation values and the baseline prompt field."""
        return dict(prompt=instruction, image=extracted_obs["image"])

    def _query_server(self, request):
        """Use the transport stub through the same hook as the real client."""
        return self._infer_with_retry(request)

    def _infer_with_retry(self, request):
        """Record metadata and return deterministic actions/reset acknowledgements."""
        self.requests.append(dict(request))
        if request.get("c3ache_control") == "reset":
            if self.fail_reset:
                raise OSError("simulated reset delivery failure")
            return {"reset": True}
        return {"action": np.ones((2, 8)), "c3ache": {"cache_hits": 4}}

    def _unpack_response(self, response):
        """Read baseline-compatible action arrays."""
        return response["action"]


stub = ModuleType("policies.cosmos3.client")
stub.Cosmos3Client = BaselineClient
sys.modules[stub.__name__] = stub
client_module = load_file("c3ache_client_under_test", ROOT / "integrations/robolab/c3ache_client.py")


class ClientTests(unittest.TestCase):
    """Per-env identity, chunk counters, task switches and reset delivery."""

    def setUp(self):
        """Connect to an in-memory transport only."""
        self.client = client_module.C3acheCosmos3Client()

    def infer(self, env_id=0, task="task"):
        """Execute one CPU action step through the real base-client loop."""
        return self.client.infer({"image": "new observation"}, task, env_id=env_id)

    def test_ids_advance_only_for_planning_chunks(self):
        """Local action playback does not advance remote chunk identifiers."""
        self.infer()
        self.infer()
        self.assertEqual(len(self.client.requests), 1)
        self.infer()
        self.assertEqual([x["chunk_id"] for x in self.client.requests], [0, 1])
        self.assertEqual(self.client.requests[0]["episode_id"], self.client.requests[1]["episode_id"])

    def test_env_and_client_sessions_are_distinct(self):
        """Neither vectorized environments nor independent client processes share keys."""
        self.infer(0)
        self.infer(1)
        self.assertNotEqual(self.client.requests[0]["session_id"], self.client.requests[1]["session_id"])
        peer = client_module.C3acheCosmos3Client()
        self.assertNotEqual(self.client._session_id, peer._session_id)

    def test_reset_notifies_server_and_preserves_other_env(self):
        """Resetting env 0 rotates its episode while env 1 retains its chunk."""
        self.infer(0)
        self.infer(1)
        first = self.client.requests[0]
        self.client.reset(env_id=0)
        reset = self.client.requests[-1]
        self.assertEqual(reset["c3ache_control"], "reset")
        self.assertEqual(reset["episode_id"], first["episode_id"])
        self.assertIn(1, self.client._chunks)
        self.infer(0)
        self.assertEqual(self.client.requests[-1]["chunk_id"], 0)
        self.assertNotEqual(self.client.requests[-1]["episode_id"], first["episode_id"])

    def test_lost_reset_still_rotates_episode(self):
        """Network failure cannot make the next trial reuse the old episode key."""
        self.infer()
        first = self.client.requests[0]
        self.client.fail_reset = True
        with self.assertLogs(client_module.logger, level="WARNING"):
            self.client.reset()
        self.infer()
        self.assertNotEqual(self.client.requests[-1]["episode_id"], first["episode_id"])

    def test_task_switch_discards_unexecuted_chunk(self):
        """New instructions immediately trigger reset and fresh planning."""
        self.infer(task="one")
        first = self.client.requests[0]
        self.infer(task="two")
        self.assertEqual(self.client.requests[-1]["prompt"], "two")
        self.assertNotEqual(self.client.requests[-1]["episode_id"], first["episode_id"])


def load_server_infer():
    """Compile the actual protocol entry without loading model/server dependencies."""
    tree = ast.parse((ROOT / "cosmos_framework/scripts/action_policy_server_robolab.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RobolabPolicyService")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "infer")
    from dataclasses import asdict

    ns = dict(Any=object, C3acheCache=C3acheCache, asdict=asdict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "server-infer", "exec"), ns)
    return ns["infer"]


class ServerProtocolTests(unittest.TestCase):
    """Reset never consumes RNG/model work; disabled mode keeps its legacy response."""

    def test_reset_does_not_generate_and_disabled_client_is_compatible(self):
        """Test the real server entry's disabled/reset branches with missing action data."""
        calls = []
        service = SimpleNamespace(_lock=threading.RLock(), _c3ache_cache=None, _infer_impl=calls.append)
        infer = load_server_infer()
        self.assertEqual(infer(service, dict(c3ache_control="reset", session_id="s", episode_id="e")), {"reset": True})
        self.assertEqual(calls, [])
        infer(service, {"prompt": "legacy client"})
        self.assertEqual(calls, [{"prompt": "legacy client"}])

    def test_invalid_enabled_request_fails_before_rng_or_model(self):
        """Enabled caching requires identities rather than guessing from the task string."""
        cfg_type = make_dataclass("Config", [("guidance", float)])
        calls = []
        service = SimpleNamespace(
            _lock=threading.RLock(),
            _c3ache_cache=C3acheCache(C3acheConfig()),
            _c3ache_sampler="unipc",
            cfg=cfg_type(3.0),
            _infer_impl=calls.append,
            model=SimpleNamespace(net=SimpleNamespace(language_model=SimpleNamespace(model=object()))),
        )
        with self.assertRaises(ValueError):
            load_server_infer()(service, {"prompt": "no identity"})
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
