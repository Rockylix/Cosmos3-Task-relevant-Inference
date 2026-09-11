"""CPU-only behavioral tests; execute real edited stack code with tiny fake layers.

No checkpoints, simulators, CUDA operations or heavy inference imports are used.
"""

import ast
import importlib.util
import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT.parent.parent / "cosmos-framework-edge-baseline"


def load_file(name, path):
    """Import a dependency-light source file without loading the inference stack."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cache_module = load_file("c3ache_under_test", ROOT / "cosmos_framework/inference/c3ache.py")
C3acheCache, C3acheConfig = cache_module.C3acheCache, cache_module.C3acheConfig


def get_device_and_dtype(pack):
    """Return the CPU tensor placement used by the real stack body."""
    value = pack["full_only_seq"]
    return value.device, value.dtype


def from_all_seq(value, pack):
    """Provide positional metadata to the tiny decoder without real attention."""
    return value


def zeros_like(pack):
    """Allocate final-norm outputs with the same two-pathway structure."""
    return {key: torch.zeros_like(value) for key, value in pack.items()}


def load_stack(root):
    """Compile the actual _impl_forward body, avoiding optional GPU dependencies."""
    path = root / "cosmos_framework/model/generator/mot/unified_mot.py"
    tree = ast.parse(path.read_text())
    stack = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_impl_forward")
    helpers_path = root / "cosmos_framework/data/generator/sequence_packing/runtime.py"
    helpers = ast.parse(helpers_path.read_text())
    names = {"get_gen_seq", "set_gen_seq", "get_und_seq", "set_und_seq"}
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body += [node for node in helpers.body if isinstance(node, ast.FunctionDef) and node.name in names]
    body.append(stack)
    namespace = dict(
        torch=torch, get_device_and_dtype=get_device_and_dtype, from_all_seq=from_all_seq, zeros_like=zeros_like
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace["_impl_forward"]


class TinyLayer:
    """Nonlinear layer whose residual changes with current input and branch/step."""

    def __init__(self, owner, index):
        """Record invocation counts for whole-stack skip assertions."""
        self.owner, self.index = owner, index

    def __call__(self, pack, *args, **kwargs):
        """Apply distinct per-branch transforms, optionally mutating the input."""
        self.owner.layer_calls += 1
        result = dict(pack)
        value = pack["full_only_seq"]
        if self.owner.mutate:
            value.add_(self.index + self.owner.branch + self.owner.step + 1)
        result["full_only_seq"] = value * 1.25 + self.owner.branch + self.owner.step + self.index
        return result, {}, None


class TinyTransformer:
    """Minimal host of the actual baseline/edited Transformer loop."""

    def __init__(self, mutate=False):
        """Create three layers and a deliberately non-identity final norm."""
        self.training = False
        self.layer_calls = self.norm_calls = 0
        self.branch = self.step = 0
        self.mutate = mutate
        self.layers = [TinyLayer(self, index) for index in range(3)]

    def rotary_emb(self, tensor, position_ids):
        """Supply deterministic fake RoPE metadata for the real loop."""
        return position_ids.to(dtype=tensor.dtype), position_ids.to(dtype=tensor.dtype)

    def norm(self, value):
        """Keep the unused understanding path unchanged."""
        return value

    def norm_moe_gen(self, value):
        """Make a wrong post-norm residual boundary numerically detectable."""
        self.norm_calls += 1
        return value * 3 + 7


def packed_layout(mask=1, shape=(2, 1, 2)):
    """Include condition and action positions with geometry-changing variants."""
    modality = SimpleNamespace(
        token_shapes=[shape],
        sequence_indexes=torch.arange(4),
        condition_mask=[torch.tensor([mask, 0])],
        domain_id=[],
        raw_action_dim=None,
    )
    action = SimpleNamespace(
        token_shapes=[(2,)],
        sequence_indexes=torch.arange(4, 6),
        condition_mask=[torch.tensor([1, 0])],
        domain_id=[torch.tensor(0)],
        raw_action_dim=None,
    )
    return SimpleNamespace(
        sample_lens=[7],
        split_lens=[1, 6],
        attn_modes=["causal", "full"],
        text_ids=torch.tensor([42]),
        text_indexes=torch.tensor([0]),
        position_ids=torch.arange(7),
        num_action_tokens_per_supertoken=0,
        null_action_supertokens=0,
        vision=modality,
        action=action,
        sound=None,
    )


class CacheTests(unittest.TestCase):
    """Numerical boundary, schedule and episode-isolation regression tests."""

    def setUp(self):
        """Load actual changed/baseline stack bodies and a fresh tiny transformer."""
        self.stack = load_stack(ROOT)
        self.baseline_stack = load_stack(BASELINE)
        self.model, self.net = TinyTransformer(), SimpleNamespace()
        self.cache = C3acheCache(C3acheConfig(refresh_period=4))

    def run_chunk(
        self,
        chunk,
        *,
        session="s",
        episode="e",
        signature="task-a",
        layout=None,
        timestep_shift=0,
        branches=("conditional", "unconditional"),
        dtype=torch.float64,
    ):
        """Run four steps through actual edited stack code and return outputs/counts."""
        obs = dict(session_id=session, episode_id=episode, chunk_id=chunk)
        outputs = {}
        with (
            torch.inference_mode(),
            self.cache.request(
                obs, signature=signature, transformer=self.model, net=self.net, branches=branches
            ) as request,
        ):
            for step in range(4):
                for branch in branches:
                    self.model.step = step
                    self.model.branch = 10 if branch == "unconditional" else 0
                    h0 = torch.arange(12, dtype=dtype).reshape(6, 2) + chunk
                    with request.forward_context(step, branch, torch.tensor([[step + timestep_shift]])):
                        request.set_layout(layout or packed_layout(), False)
                        result, _ = self.stack(
                            self.model,
                            {"causal_seq": torch.ones(1, 2, dtype=dtype), "full_only_seq": h0},
                            None,
                            torch.arange(7),
                        )
                        outputs[(branch, step)] = result["full_only_seq"].clone()
        return outputs, request

    def test_disabled_and_period_one_equal_baseline(self):
        """Disabled and always-refresh paths preserve exact dense outputs in two dtypes."""
        self.cache = C3acheCache(C3acheConfig(refresh_period=1))
        for dtype in (torch.float64, torch.bfloat16):
            for chunk in (0, 1):
                outputs, request = self.run_chunk(chunk, dtype=dtype)
                self.assertEqual(request.stats["cache_hits"], 0)
                for (branch, step), output in outputs.items():
                    model = TinyTransformer()
                    model.step, model.branch = step, 10 if branch == "unconditional" else 0
                    pack = {
                        "causal_seq": torch.ones(1, 2, dtype=dtype),
                        "full_only_seq": torch.arange(12, dtype=dtype).reshape(6, 2) + chunk,
                    }
                    with torch.inference_mode():
                        expected, _ = self.baseline_stack(model, pack, None, torch.arange(7))
                        disabled, _ = self.stack(model, pack, None, torch.arange(7))
                    self.assertTrue(torch.equal(output, expected["full_only_seq"]))
                    self.assertTrue(torch.equal(disabled["full_only_seq"], expected["full_only_seq"]))

    def test_entire_stack_skipped_and_norm_stays_live(self):
        """Reuse pre-norm residuals on both early steps/all GEN rows and both CFG branches."""
        first, _ = self.run_chunk(0)
        count = self.model.layer_calls
        second, request = self.run_chunk(1)
        self.assertEqual(request.stats["cache_hits"], 4)
        self.assertEqual(self.model.layer_calls - count, 4 * len(self.model.layers))
        self.assertEqual(self.model.norm_calls, 16)
        for branch in ("conditional", "unconditional"):
            for step in (0, 1):
                # New h0 differs by one, so the LIVE final norm changes output by three.
                self.assertTrue(torch.equal(second[(branch, step)], first[(branch, step)] + 3))
            self.assertFalse(torch.equal(second[(branch, 2)], first[(branch, 2)] + 3))
        self.assertIsNone(self.net._c3ache_request)
        self.assertIsNone(self.model._c3ache_request)

    def test_refresh_periods_and_protected_tail(self):
        """Exercise requested periods 2/4/8 and both CCDD/CCCD schedules."""
        for period in (2, 4, 8):
            for tail in (1, 2):
                self.cache = C3acheCache(C3acheConfig(refresh_period=period, dense_tail_steps=tail))
                for chunk in range(period + 1):
                    _, request = self.run_chunk(chunk)
                    self.assertEqual(request.stats["cache_hits"], 0 if chunk % period == 0 else 2 * (4 - tail))

    def test_single_cfg_branch(self):
        """Guidance=1 uses explicit conditional tags without assuming alternating calls."""
        self.run_chunk(0, branches=("conditional",))
        _, request = self.run_chunk(1, branches=("conditional",))
        self.assertEqual(request.stats["cache_hits"], 2)

    def test_actual_sampler_velocity_closure_tags_branches_and_steps(self):
        """Run the edited velocity closure itself, with only the heavy network replaced."""
        path = ROOT / "cosmos_framework/model/generator/omni_mot_model.py"
        tree = ast.parse(path.read_text())
        model_cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "OmniMoTModel")
        generate = next(
            node
            for node in model_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate_samples_from_batch"
        )
        velocity = next(
            node for node in ast.walk(generate) if isinstance(node, ast.FunctionDef) and node.name == "velocity_fn"
        )
        factory = ast.parse("def factory():\n    c3ache_step = -1\n    uncond_text_kv_cache = None\n").body[0]
        factory.body.extend([velocity, ast.Return(value=ast.Name(id="velocity_fn", ctx=ast.Load()))])
        cfg_fn = next(
            node
            for node in model_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_classifier_free_guidance"
        )
        cond_tokens, uncond_tokens = [[12]], [[]]
        seen = []

        def get_velocity(**kwargs):
            """Observe active tags supplied by real closure and complete a tiny stack call."""
            key = request.active[0]
            seen.append(key)
            request.set_layout(packed_layout(), False)
            self.stack(
                self.model, {"full_only_seq": torch.ones(6, 2), "causal_seq": torch.ones(1, 2)}, None, torch.arange(7)
            )
            return [torch.ones(1)]

        host = SimpleNamespace(parallel_dims=None, _get_velocity=get_velocity)
        namespace = dict(
            torch=torch,
            nullcontext=nullcontext,
            self=host,
            reuse_pack_templates=False,
            reuse_text_kv=False,
            net=None,
            has_noisy_actions=True,
            sequence_plans=None,
            gen_data_clean=None,
            cond_tokens=cond_tokens,
            uncond_tokens=uncond_tokens,
            skip_text_tokens_for_cfg=False,
            guidance=3.0,
            guidance_interval=None,
            _dp_shard_group=None,
            velocity_postprocess=None,
            normalize_cfg=False,
        )
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cfg_fn, factory],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
        from types import MethodType

        host._run_classifier_free_guidance = MethodType(namespace["_run_classifier_free_guidance"], host)
        for chunk in (0, 1):
            with (
                torch.inference_mode(),
                patch("torch.compiler.cudagraph_mark_step_begin"),
                self.cache.request(
                    dict(session_id="s", episode_id="e", chunk_id=chunk),
                    signature="task-a",
                    transformer=self.model,
                    net=self.net,
                    branches=("conditional", "unconditional"),
                ) as request,
            ):
                namespace["c3ache_request"] = request
                velocity_fn = namespace["factory"]()
                for step in range(4):
                    velocity_fn([torch.ones(1)], torch.tensor([[step]]))
            self.assertEqual(request.stats["cache_hits"], 0 if chunk == 0 else 4)
        expected = [(branch, step) for step in range(4) for branch in ("conditional", "unconditional")]
        self.assertEqual(seen, expected * 2)

    def test_in_place_layer_does_not_corrupt_h0(self):
        """The dense residual owns its input snapshot even if a decoder mutates the pack."""
        self.model = TinyTransformer(mutate=True)
        first, _ = self.run_chunk(0)
        second, _ = self.run_chunk(1)
        self.assertTrue(torch.equal(first[("conditional", 0)] + 3, second[("conditional", 0)]))

    def test_task_episode_session_and_chunk_discontinuities(self):
        """Same prompt across trials never shares residuals; repeats/skips force refresh."""
        self.run_chunk(0)
        for chunk, options in (
            (1, {"episode": "new"}),
            (1, {"session": "other"}),
            (1, {"signature": "new-task"}),
            (0, {}),
            (3, {}),
        ):
            _, request = self.run_chunk(chunk, **options)
            self.assertEqual(request.stats["cache_hits"], 0)

    def test_layout_masks_dtype_and_timestep_invalidate(self):
        """Reject same-size geometry/mask changes and reuse at the wrong time value."""
        for options in (
            {"layout": packed_layout(mask=0)},
            {"layout": packed_layout(shape=(2, 2, 1))},
            {"timestep_shift": 1},
            {"dtype": torch.bfloat16},
        ):
            self.cache = C3acheCache(C3acheConfig(refresh_period=4))
            self.run_chunk(0)
            _, request = self.run_chunk(1, **options)
            self.assertEqual(request.stats["cache_hits"], 0)
            self.assertEqual(request.reason, "layout_or_timestep_changed")

    def test_reset_isolation_and_lru_bound(self):
        """Reset one environment without discarding peers; evicted state starts dense."""
        self.cache = C3acheCache(C3acheConfig(max_sessions=2, refresh_period=4))
        self.run_chunk(0, session="one")
        self.run_chunk(0, session="two")
        self.cache.reset(dict(session_id="one", episode_id="e"))
        _, peer = self.run_chunk(1, session="two")
        self.assertEqual(peer.stats["cache_hits"], 4)
        self.run_chunk(0, session="three")
        self.run_chunk(0, session="four")
        self.assertEqual(len(self.cache.episodes), 2)
        _, evicted = self.run_chunk(2, session="two")
        self.assertEqual(evicted.stats["cache_hits"], 0)

    def test_exception_and_incomplete_schedule_drop_state(self):
        """Failures restore model attributes and cannot commit partial residuals."""
        for fail in (True, False):
            with self.assertRaises(RuntimeError):
                with self.cache.request(
                    dict(session_id="s", episode_id="e", chunk_id=0),
                    signature="task-a",
                    transformer=self.model,
                    net=self.net,
                    branches=("conditional",),
                ):
                    if fail:
                        raise RuntimeError("simulated decode failure")
            self.assertFalse(self.cache.episodes)
            self.assertIsNone(self.net._c3ache_request)

    def test_missing_identity_and_invalid_chunk_rejected(self):
        """An old client cannot silently share the enabled cache with other episodes."""
        for obs in ({}, {"session_id": "s", "episode_id": "e", "chunk_id": True}):
            with self.assertRaises(ValueError):
                with self.cache.request(
                    obs, signature="a", transformer=self.model, net=self.net, branches=("conditional",)
                ):
                    self.fail("invalid request accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
