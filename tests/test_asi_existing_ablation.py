import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_framework.inference import asi_existing_ablation as ablation
from cosmos_framework.inference import edge_core_stable as optimized
from cosmos_framework.inference import edge_core_stable_fast as fast
from cosmos_framework.scripts import robolab_version1 as legacy


def reference_attention(*, query, key, value, scale, return_lse):
    heads, kv_heads = query.shape[2], key.shape[2]
    q = query.float().transpose(1, 2)
    k = key.float().repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2)
    v = value.float().repeat_interleave(heads // kv_heads, dim=2).transpose(1, 2)
    logits = (q * scale) @ k.transpose(-1, -2)
    return (logits.softmax(-1) @ v).transpose(1, 2), logits.logsumexp(-1).transpose(1, 2)


def net_fixture():
    layers = [SimpleNamespace(self_attn=SimpleNamespace(_attention_stats_capture_callback=None)) for _ in range(28)]
    return SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))


def controller(mode):
    return ablation.CONTROLLERS[mode](torch=torch, net=net_fixture(), guidance=3, num_steps=4)


def layout_fixture(spatial=7):
    return {
        "num_gen_tokens": 9 * spatial + 33,
        "latent_positions": {f"L{f}": list(range(f * spatial, (f + 1) * spatial)) for f in range(9)},
        "action_queries": [
            dict(query_role="predicted", action_horizon=h, gen_position=9 * spatial + 1 + h) for h in range(32)
        ],
    }


class ExistingAblationTest(unittest.TestCase):
    def test_only_requested_mechanisms_enabled(self):
        b2, b3, b4, b5 = [controller(f"b{i}") for i in range(2, 6)]
        for c in (b2, b3):
            self.assertIs(c.run_layer.__func__, legacy.Version1Controller.run_layer)
            self.assertIs(c._slice_pack_and_rope.__func__, legacy.Version1Controller._slice_pack_and_rope)
            self.assertIs(c.end_stack.__func__, legacy.Version1Controller.end_stack)
        self.assertTrue(b2.validate_each_profile)
        self.assertFalse(b3.validate_each_profile)
        for c in (b4, b5):
            self.assertIs(c._slice_pack_and_rope.__func__, optimized.Version1Controller._slice_pack_and_rope)
            self.assertFalse(c.compile_profile_kernel)
            self.assertFalse(c.cuda_graphs)
            self.assertTrue(c.cache_layout)
            self.assertTrue(c.optimized)
        self.assertFalse(b4.compile_profile_decoder)
        self.assertTrue(b5.compile_profile_decoder)

    def test_layout_identity_cache_and_single_geometry_build(self):
        for mode in ablation.CONTROLLERS:
            c = controller(mode)
            layout = layout_fixture()
            first, second = object(), object()
            with (
                patch.object(optimized, "_action_query_layout", return_value=layout) as resolve,
                patch.object(fast, "profile_geometry", wraps=fast.profile_geometry) as geometry,
            ):
                for packed in (first, first, second, second):
                    c._network_pre_hook(None, (packed,), {})
                self.assertEqual(resolve.call_count, 2)
                self.assertEqual(geometry.call_count, 1)
            self.assertIs(c._layout_cache[id(first)][0], first)
            self.assertIs(c._layout_cache[id(second)][0], second)
            self.assertEqual(controller(mode)._layout_cache, {})

    def test_changed_layout_rejected_for_new_object(self):
        c = controller("b2")
        first = layout_fixture()
        second = layout_fixture(spatial=8)
        with patch.object(optimized, "_action_query_layout", side_effect=[first, second]):
            c._network_pre_hook(None, (object(),), {})
            with self.assertRaisesRegex(RuntimeError, "layout changed"):
                c._network_pre_hook(None, (object(),), {})

    def test_fast_score_matches_legacy_and_full_key_normalizer(self):
        torch.manual_seed(57)
        layout = layout_fixture()
        size = layout["num_gen_tokens"]
        inputs = dict(
            q_gen=torch.randn(size, 16, 16),
            k_ar=torch.randn(11, 8, 16),
            k_gen=torch.randn(size, 8, 16),
            v_ar=torch.randn(11, 8, 16),
            v_gen=torch.randn(size, 8, 16),
            scaling=0.25,
        )
        with (
            patch.object(legacy, "attention", side_effect=reference_attention),
            patch.object(fast, "attention", side_effect=reference_attention),
        ):
            expected = legacy.action_aligned_future_profiles_with_lse(torch=torch, token_layout=layout, **inputs)
            actual = fast.action_aligned_future_profiles(geometry=fast.profile_geometry(layout), **inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        self.assertTrue(bool((actual.sum(-1) < 1).all()))

    def test_b2_finite_immediate_b3_deferred_to_unchanged_planner(self):
        profile = torch.full((8, 340), float("nan"))
        for mode in ("b2", "b3", "b4"):
            c = controller(mode)
            c._profile_callback_block = 0
            c._layout = layout_fixture(340)
            c._profile_geometry = fast.profile_geometry(c._layout)
            inputs = {k: None for k in ("q_gen", "k_ar", "k_gen", "v_ar", "v_gen")}
            with patch.object(fast, "action_aligned_future_profiles", return_value=profile):
                if mode == "b2":
                    with self.assertRaisesRegex(RuntimeError, "profile is invalid"):
                        c._capture_profile(layer_index=0, scaling=1, **inputs)
                else:
                    c._capture_profile(layer_index=0, scaling=1, **inputs)
                    self.assertEqual(len(c._profile_records), 1)
        for planner in (legacy.build_core_stable_plan, optimized.build_core_stable_plan):
            with self.assertRaisesRegex(RuntimeError, "NaN/Inf"):
                planner(torch=torch, profile_records=[dict(block=b, profiles=profile) for b in range(28)])

    def test_production_and_legacy_planners_identical_budgets(self):
        torch.manual_seed(44)
        records = [dict(block=b, profiles=torch.rand(8, 340)) for b in range(28)]
        a = legacy.build_core_stable_plan(torch=torch, profile_records=records)
        b = optimized.build_core_stable_plan(torch=torch, profile_records=records)
        self.assertEqual(a["core_blocks"], b["core_blocks"])
        for field in ("execution_mask", "core_masks", "stable_mask", "block_quality"):
            self.assertTrue(torch.equal(a[field], b[field]))
        self.assertEqual(a["core_masks"].sum(-1).tolist(), [80] * 8)
        self.assertEqual(a["execution_mask"].sum(-1).tolist(), [184] * 8)
        self.assertEqual(int(a["stable_mask"].sum()), 104)

    def test_sparse_metadata_cache_preserves_current_values_and_rope(self):
        c = controller("b4")
        selected = torch.tensor([0, 2, 6])
        und = torch.randn(4, 2)
        original = optimized._make_sequence_pack(und_seq=und, gen_seq=torch.randn(8, 2))
        cos = optimized._make_sequence_pack(und_seq=und, gen_seq=torch.randn(8, 2))
        sin = optimized._make_sequence_pack(und_seq=und, gen_seq=torch.randn(8, 2))
        c._position_embeddings = (cos, sin)
        first, rope = c._slice_pack_and_rope(original, selected)
        newer = {**original, "full_only_seq": torch.randn(8, 2)}
        with patch.object(optimized, "_make_sequence_pack", wraps=optimized._make_sequence_pack) as make:
            second, _ = c._slice_pack_and_rope(newer, selected)
            self.assertEqual(make.call_count, 0)
        self.assertEqual(len(c._sparse_metadata), 1)
        self.assertTrue(torch.equal(second["full_only_seq"], newer["full_only_seq"].index_select(0, selected)))
        self.assertFalse(torch.equal(first["full_only_seq"], second["full_only_seq"]))
        for result, source in zip(rope, (cos, sin), strict=True):
            self.assertTrue(torch.equal(result["full_only_seq"], source["full_only_seq"].index_select(0, selected)))

    def test_metadata_route_not_callback_and_no_compile(self):
        c = controller("b5")
        c._profile_geometry = (8, 2, 7)
        c._position_embeddings = (None, None)
        profile = torch.rand(8, 7)
        calls = []

        def decoder(*args, **kwargs):
            calls.append(kwargs)
            return "output", {"asi_profile": profile, "other": 7}, "kv"

        with patch.object(torch, "compile", side_effect=AssertionError("unexpected compile")):
            output, metadata, kv = c._run_dense_profile_layer(
                block=0,
                decoder_layer=decoder,
                hidden_states=None,
                attention_mask=None,
                memory_value=None,
                gen_only=True,
            )
        self.assertEqual((output, metadata, kv), ("output", {"other": 7}, "kv"))
        self.assertEqual(calls[0]["asi_profile_geometry"], c._profile_geometry)
        self.assertTrue(torch.equal(c._profile_records[0]["profiles"], profile))
        self.assertNotEqual(c._profile_records[0]["profiles"].data_ptr(), profile.data_ptr())


if __name__ == "__main__":
    unittest.main()
