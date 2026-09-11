"""CPU contract tests; no model weights, GPU inference, or extra environment.

Run: PYTHONPATH=. <project-python> -m unittest discover -s tests -p worldcache_dddc_test.py -v
"""

from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

from cosmos_framework.inference.worldcache import (
    OutputLayout,
    WorldCacheConfig,
    WorldCacheRequest,
    curvature_and_slopes,
    heterogeneous_predict,
    patchify,
    unpatchify,
)

ROOT = Path(__file__).resolve().parents[1]


def packed_fixture(dtype=torch.float32, height=4, width=6):
    return NS(
        vision=NS(
            tokens=[torch.zeros(2, 9, height, width, dtype=dtype)],
            condition_mask=[torch.tensor([1.0] + [0.0] * 8).reshape(9, 1, 1)],
        ),
        action=NS(
            tokens=[torch.zeros(33, 6, dtype=dtype)], condition_mask=[torch.tensor([1.0] + [0.0] * 32).reshape(33, 1)]
        ),
        sound=None,
    )


def prediction(packed, step, branch):
    # Nonuniform token/channel trajectories, both modalities and branch histories distinct.
    bias = 11 if branch == "conditional" else -7
    result = {}
    for key in ("vision", "action"):
        t = getattr(packed, key).tokens[0]
        base = torch.arange(t.numel()).reshape(t.shape).float() / t.numel()
        result[f"preds_{key}"] = [(base + bias + 0.2 * step + 0.04 * step * step * base).to(t.dtype)]
    return result


def extracted_velocity_tail():
    """Execute the ACTUAL model dispatch/masking/flatten code without loading its heavyweight class.

    This is a CPU seam test, not a neural-network inference test.
    """
    tree = ast.parse((ROOT / "cosmos_framework/model/generator/omni_mot_model.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_get_velocity")
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "worldcache_request is None"
    )
    fn = ast.parse("def run_tail(): pass").body[0]
    fn.body = copy.deepcopy(method.body[start:])
    code = compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "actual_velocity_tail", "exec")
    return code


class WorldCacheTests(unittest.TestCase):
    def test_patch_grid_and_exact_network_order(self):
        x = torch.arange(3 * 8 * 34 * 40).reshape(3, 8, 34, 40).float()
        y = patchify(x, 2)
        self.assertEqual(y.shape, (8 * 340, 12))
        expected = torch.einsum("cthpwq->thwpqc", x.reshape(3, 8, 17, 2, 20, 2)).reshape(-1, 12)
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        torch.testing.assert_close(unpatchify(y, tuple(x.shape), 2), x, rtol=0, atol=0)

    def test_layout_restores_future_and_zeros_conditions_padding(self):
        p = packed_fixture()
        layout = OutputLayout.from_packed(p, 2, 4)
        out = prediction(p, 0, "conditional")
        restored = layout.decode(layout.encode(out))
        torch.testing.assert_close(restored["preds_vision"][0][:, 1:], out["preds_vision"][0][:, 1:])
        torch.testing.assert_close(restored["preds_action"][0][1:, :4], out["preds_action"][0][1:, :4])
        self.assertEqual(int(torch.count_nonzero(restored["preds_vision"][0][:, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(restored["preds_action"][0][0])), 0)
        self.assertEqual(int(torch.count_nonzero(restored["preds_action"][0][:, 4:])), 0)

    def test_layout_rejects_unsupported_conditioning_and_padding(self):
        p = packed_fixture()
        p.vision.condition_mask[0][2] = 1
        with self.assertRaises(ValueError):
            OutputLayout.from_packed(p, 2, 4)
        p = packed_fixture(height=33, width=40)
        layout = OutputLayout.from_packed(p, 2, 4)
        with self.assertRaisesRegex(ValueError, "native llm2vae"):
            layout.encode(prediction(p, 0, "conditional"))

    def test_native_padded_projection_history_and_crop(self):
        p = packed_fixture(height=33, width=40)
        native = torch.randn(2, 8, 34, 40)
        native[:, :, 33, :] = 123  # True projected padding, not guessed zero padding.
        projection = torch.nn.Identity()
        req = WorldCacheRequest(WorldCacheConfig())
        req.begin_step()
        out = prediction(p, 0, "conditional")
        out["preds_vision"][0][:, 1:] = native[:, :, :33]

        def compute():
            projection(patchify(native, 2))
            return out

        got = req.evaluate("conditional", compute, p, 2, 4, projection)
        stored = req.histories["conditional"].outputs[0]
        torch.testing.assert_close(stored["vision"], patchify(native, 2), rtol=0, atol=0)
        restored = req.histories["conditional"].layout.decode(stored)
        torch.testing.assert_close(restored["preds_vision"][0][:, 1:], got["preds_vision"][0][:, 1:])
        self.assertFalse(projection._forward_hooks)

    def test_native_batched_vision_shape(self):
        p = packed_fixture(height=33, width=40)
        p.vision.tokens[0] = p.vision.tokens[0].unsqueeze(0)
        native = torch.randn(2, 8, 34, 40)
        out = prediction(p, 0, "conditional")
        out["preds_vision"][0][0, :, 1:] = native[:, :, :33]
        layout = OutputLayout.from_packed(p, 2, 4)
        restored = layout.decode(layout.encode(out, patchify(native, 2)))
        self.assertEqual(restored["preds_vision"][0].shape, (1, 2, 9, 33, 40))
        torch.testing.assert_close(restored["preds_vision"][0][0, :, 1:], native[:, :, :33])
        projection = torch.nn.Identity()
        req2 = WorldCacheRequest(WorldCacheConfig())
        req2.begin_step()
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            req2.evaluate(
                "conditional", lambda: (_ for _ in ()).throw(RuntimeError("test failure")), p, 2, 4, projection
            )
        self.assertFalse(projection._forward_hooks)

    def test_matches_actual_network_unpatchify_with_native_padding(self):
        path = ROOT / "cosmos_framework/model/generator/mot/cosmos3_vfm_network.py"
        tree = ast.parse(path.read_text())
        fn = copy.deepcopy(
            next(
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "unpatchify_and_unpack_latents"
            )
        )
        fn.decorator_list = []
        source = "from __future__ import annotations\n" + ast.unparse(fn)
        namespace = {"torch": torch}
        exec(compile(source, str(path), "exec"), namespace)
        p = packed_fixture(height=33, width=40, dtype=torch.bfloat16)
        p.vision.tokens[0] = torch.zeros(1, 48, 9, 33, 40, dtype=torch.bfloat16)
        layout = OutputLayout.from_packed(p, 2, 4)
        native_tokens = torch.randn(2720, 192, dtype=torch.bfloat16)
        actual = namespace[fn.name](
            NS(latent_patch_size=2, latent_channel=48),
            native_tokens,
            [(9, 17, 20)],
            [torch.arange(1, 9)],
            [(9, 33, 40)],
        )[0]
        restored = layout.decode({"vision": native_tokens, "action": torch.zeros(32, 4)})["preds_vision"][0]
        torch.testing.assert_close(restored, actual, rtol=0, atol=0)

    def test_curvature_uses_full_step_index_gaps(self):
        steps = [0, 2, 5]
        history = [torch.tensor([[s * s, 2 * s]], dtype=torch.float32) for s in steps]
        curve, curr, prev = curvature_and_slopes(history, steps, 1e-8)
        torch.testing.assert_close(prev, torch.tensor([[2.0, 2.0]]))
        torch.testing.assert_close(curr, torch.tensor([[7.0, 2.0]]))
        torch.testing.assert_close(curve, torch.tensor([(5 / 3) / 53]))

    def test_three_prediction_rules(self):
        cfg = WorldCacheConfig()
        latest = torch.ones(3, 2)
        curr, prev = torch.full((3, 2), 3.0), torch.full((3, 2), -2.0)
        curve, thresholds = torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.5, 1.5])
        got = heterogeneous_predict(latest, curr, prev, curve, thresholds, cfg)
        alpha = 3 * (1 / 6) ** 2 - 2 * (1 / 6) ** 3
        torch.testing.assert_close(got[0], torch.ones(2))
        torch.testing.assert_close(got[1], torch.full((2,), 4.0))
        torch.testing.assert_close(got[2], torch.full((2,), 1 + 3 * (1 - alpha) - 2 * alpha))

    def test_constant_history_and_quantile_ties_finite(self):
        h = [torch.ones(8, 12)] * 3
        c, curr, prev = curvature_and_slopes(h, [0, 1, 2], 1e-8)
        out = heterogeneous_predict(h[-1], curr, prev, c, torch.zeros(2), WorldCacheConfig())
        torch.testing.assert_close(out, h[-1], rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_dddc_six_real_forwards_branch_isolation_and_no_alias(self):
        cfg = WorldCacheConfig()
        req = WorldCacheRequest(cfg)
        p = packed_fixture(dtype=torch.bfloat16)
        full_calls, cached = [], {}
        for step in range(4):
            req.begin_step()
            # Branch order is not assumed by the controller.
            for branch in reversed(req.branches):

                def compute():
                    full_calls.append((step, branch))
                    return prediction(p, step, branch)

                got = req.evaluate(branch, compute, p, 2, 4)
                if step < 3:
                    torch.testing.assert_close(got["preds_vision"][0], prediction(p, step, branch)["preds_vision"][0])
                    got["preds_vision"][0].zero_()  # History must not alias caller outputs.
                else:
                    cached[branch] = got
        self.assertEqual(len(full_calls), 6)
        self.assertTrue(all(s < 3 for s, b in full_calls))
        self.assertFalse(
            torch.equal(cached["conditional"]["preds_vision"][0], cached["unconditional"]["preds_vision"][0])
        )
        for branch in req.branches:
            self.assertEqual(req.histories[branch].steps, [0, 1, 2])
            self.assertNotEqual(int(torch.count_nonzero(req.histories[branch].outputs[0]["vision"])), 0)
        report = req.finish()
        self.assertEqual(report["cache_forwards"], 2)
        for event in report["events"][-2:]:
            for counts in event["groups"].values():
                self.assertEqual(counts["stable"] + counts["linear"] + counts["chaotic"], counts["total"])
        req.clear()
        self.assertFalse(req.histories)
        self.assertFalse(WorldCacheRequest(cfg).histories)

    def test_cache_matches_explicit_per_modality_reference(self):
        cfg, p = WorldCacheConfig(), packed_fixture()
        req = WorldCacheRequest(cfg)
        for step in range(3):
            req.begin_step()
            for b in req.branches:
                req.evaluate(b, lambda: prediction(p, step, b), p, 2, 4)
        state = req.histories["conditional"]
        curves, data = [], {}
        for name in ("vision", "action"):
            c, v, prev = curvature_and_slopes([o[name] for o in state.outputs], [0, 1, 2], cfg.eps)
            data[name] = c, v, prev
            curves.append(c)
        lo, hi = torch.quantile(torch.cat(curves), torch.tensor([0.3, 0.7]))
        alpha = 3 / 36 - 2 / 216
        expected = {}
        for name, (c, v, prev) in data.items():
            values = []
            for i in range(len(c)):
                base = state.outputs[2][name][i]
                values.append(
                    base if c[i] < lo else base + ((1 - alpha) * v[i] + alpha * prev[i] if c[i] >= hi else v[i])
                )
            expected[name] = torch.stack(values)
        req.begin_step()
        got = req.evaluate("conditional", lambda: self.fail("CACHE called dense model"), p, 2, 4)
        reference = state.layout.decode(expected)
        for name in reference:
            torch.testing.assert_close(got[name][0], reference[name][0])

    def test_invalid_step_branch_history(self):
        p, req = packed_fixture(), WorldCacheRequest(WorldCacheConfig())
        with self.assertRaises(RuntimeError):
            req.evaluate("conditional", lambda: {}, p, 2, 4)
        req.begin_step()
        req.evaluate("conditional", lambda: prediction(p, 0, "conditional"), p, 2, 4)
        with self.assertRaises(RuntimeError):
            req.evaluate("conditional", lambda: {}, p, 2, 4)
        with self.assertRaises(RuntimeError):
            req.begin_step()
        with self.assertRaises(RuntimeError):
            req.finish()

    def test_nan_inf_rejected(self):
        for bad in (float("nan"), float("inf")):
            req, p = WorldCacheRequest(WorldCacheConfig()), packed_fixture()
            req.begin_step()
            out = prediction(p, 0, "conditional")
            out["preds_action"][0][1, 0] = bad
            with self.assertRaises(FloatingPointError):
                req.evaluate("conditional", lambda: out, p, 2, 4)

    def test_invalid_hyperparameters(self):
        for kw in ({"percentile_stable": 0.8}, {"percentile_chaotic": -1}, {"n_max": 0}, {"eps": float("nan")}):
            with self.assertRaises(ValueError):
                WorldCacheConfig(**kw)

    def test_actual_dispatch_masks_cfg_and_four_unipc_updates(self):
        from cosmos_framework.model.generator.diffusion.samplers.unipc import UniPCSampler

        code = extracted_velocity_tail()
        p = packed_fixture()
        initial = torch.randn(p.vision.tokens[0].numel() + p.action.tokens[0].numel())
        results = []
        for enabled in (False, True):
            req = WorldCacheRequest(WorldCacheConfig()) if enabled else None
            calls, velocities = [], []
            step_counter = [-1]

            def velocity_fn(latents, timestep):
                step_counter[0] += 1
                if req:
                    req.begin_step()
                branch_values = []
                for branch in WorldCacheRequest.branches:

                    def denoise(**kwargs):
                        calls.append((step_counter[0], branch))
                        # Constant per-branch true velocity: the surrogate must be exact.
                        return prediction(p, 0, branch)

                    namespace = dict(
                        torch=torch,
                        worldcache_request=req,
                        worldcache_branch=branch,
                        self=NS(denoise=denoise),
                        net=NS(latent_patch_size=2, llm2vae=None),
                        packed_sequence=p,
                        gen_data_clean=NS(raw_action_dim=[4]),
                        memory=None,
                        has_noisy_actions=True,
                        has_sound=False,
                        n_samples=1,
                        num_items=None,
                        sequence_plans=[NS(has_action=True, has_sound=False)],
                    )
                    exec(code, namespace)
                    branch_values.append(namespace["run_tail"]()[0])
                cond, uncond = branch_values
                guided = uncond + 3.0 * (cond - uncond)
                velocities.append(guided)
                return [guided]

            sampler = UniPCSampler(tensor_kwargs={"device": "cpu", "dtype": torch.float32})
            result = sampler(velocity_fn, [initial.clone()], num_steps=4, shift=5.0, seed=[7])[0]
            self.assertEqual(len(velocities), 4)
            self.assertEqual(len(calls), 6 if enabled else 8)
            for velocity in velocities:
                v = velocity[: p.vision.tokens[0].numel()].reshape(p.vision.tokens[0].shape)
                a = velocity[p.vision.tokens[0].numel() :].reshape(p.action.tokens[0].shape)
                self.assertTrue(bool((v[:, 0] == 0).all()))
                self.assertTrue(bool((a[0] == 0).all()))
                self.assertTrue(bool((a[:, 4:] == 0).all()))
            if req:
                req.finish()
            results.append(result)
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
