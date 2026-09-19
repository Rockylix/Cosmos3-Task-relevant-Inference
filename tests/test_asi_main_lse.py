import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_framework.inference import edge_core_stable as core
from cosmos_framework.inference import edge_core_stable_fast as fast
from cosmos_framework.inference.asi_main_lse import DecoderMetadataController, MainLSEController
from cosmos_framework.model.generator.mot import attention as mot
from cosmos_framework.model.generator.mot import inference_text_kv_memory as memory


def reference(query, key, value, is_causal=False, scale=None, return_lse=False, **kwargs):
    ratio = query.shape[2] // key.shape[2]
    q = query.float().transpose(1, 2)
    k = key.float().repeat_interleave(ratio, 2).transpose(1, 2)
    v = value.float().repeat_interleave(ratio, 2).transpose(1, 2)
    z = (q * (scale or query.shape[-1] ** -0.5)) @ k.transpose(-1, -2)
    if is_causal:
        mask = torch.ones(z.shape[-2:], dtype=torch.bool).triu(1)
        z = z.masked_fill(mask, -torch.inf)
    out = (z.softmax(-1) @ v).transpose(1, 2)
    return (out, z.logsumexp(-1).transpose(1, 2)) if return_lse else out


class MainLSETest(unittest.TestCase):
    def test_controller_exception_restores_flags(self):
        controller = MainLSEController.__new__(MainLSEController)
        existing, absent = SimpleNamespace(_asi_reuse_main_lse=False), SimpleNamespace()
        controller.layers = [SimpleNamespace(self_attn=existing), SimpleNamespace(self_attn=absent)]
        with (
            patch.object(DecoderMetadataController, "__enter__", return_value=controller),
            patch.object(DecoderMetadataController, "__exit__", return_value=None),
        ):
            with self.assertRaisesRegex(RuntimeError, "test error"):
                with controller:
                    self.assertTrue(existing._asi_reuse_main_lse)
                    self.assertTrue(absent._asi_reuse_main_lse)
                    raise RuntimeError("test error")
        self.assertFalse(existing._asi_reuse_main_lse)
        self.assertFalse(hasattr(absent, "_asi_reuse_main_lse"))

    def test_unsupported_training_and_threeway_rejected(self):
        q, k, v = self.packs()
        with torch.enable_grad(), self.assertRaisesRegex(ValueError, "single-sample inference"):
            mot.two_way_attention(q, k, v, return_gen_lse=True)
        mask = mot.SplitInfo([5, 67], ["causal", "full"], [72], 72, is_three_way=True)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "two-way"):
            mot.dispatch_attention(q, k, v, mask, return_gen_lse=True)
        with torch.inference_mode(), self.assertRaisesRegex(ValueError, "two-way"):
            memory.dispatch_attention_with_text_kv_memory(q, k, v, mask, return_gen_lse=True)

    def packs(self):
        torch.manual_seed(55)
        return [
            core._make_sequence_pack(und_seq=torch.randn(5, h, 8), gen_seq=torch.randn(67, h, 8)) for h in (4, 2, 2)
        ]

    def test_main_output_unchanged_and_normalized_und_lse(self):
        q, k, v = self.packs()
        normed = {**k, "causal_seq": k["causal_seq"] * 2}
        with torch.inference_mode(), patch.object(mot, "attention", side_effect=reference) as calls:
            plain = mot.two_way_attention(q, k, v, packed_key_states_normalized=normed)
            changed = mot.two_way_attention(q, k, v, packed_key_states_normalized=normed, return_gen_lse=True)
        self.assertEqual(calls.call_count, 4)  # Two main attentions per call; no extra scorer.
        self.assertNotIn("_asi_gen_lse", plain)
        for key in ("causal_seq", "full_only_seq"):
            self.assertTrue(torch.equal(plain[key], changed[key]))
        _, expected = reference(
            q["full_only_seq"].unsqueeze(0),
            torch.cat((normed["causal_seq"], k["full_only_seq"])).unsqueeze(0),
            torch.cat((v["causal_seq"], v["full_only_seq"])).unsqueeze(0),
            return_lse=True,
        )
        torch.testing.assert_close(changed["_asi_gen_lse"], expected, rtol=0, atol=0)

    def test_scorer_provided_lse_never_calls_extra_attention(self):
        q, k, v = self.packs()
        # L0=3, L1..8=3 each, action0=27, predicted action starts28.
        geometry = (28, 3, 3)
        args = (
            q["full_only_seq"],
            k["causal_seq"],
            k["full_only_seq"],
            v["causal_seq"],
            v["full_only_seq"],
            8**-0.5,
            geometry,
        )
        with patch.object(fast, "attention", side_effect=reference) as calls:
            expected = fast.action_aligned_future_profiles(*args)
            self.assertEqual(calls.call_count, 1)
        _, main_lse = reference(
            q["full_only_seq"].unsqueeze(0),
            torch.cat((k["causal_seq"], k["full_only_seq"])).unsqueeze(0),
            torch.cat((v["causal_seq"], v["full_only_seq"])).unsqueeze(0),
            return_lse=True,
        )
        with patch.object(fast, "attention", side_effect=AssertionError("extra attention")):
            actual = fast.action_aligned_future_profiles(*args, action_lse=main_lse[:, 28:60])
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        self.assertTrue(bool((actual.sum(-1) < 1).all()))

    def test_cached_text_lse_matches_actual_keys(self):
        q, k, v = self.packs()
        state = SimpleNamespace(
            gen_len=67, und_k_cached=k["causal_seq"].unsqueeze(0) * 3, und_v_cached=v["causal_seq"].unsqueeze(0)
        )
        with torch.inference_mode(), patch.object(memory, "attention", side_effect=reference):
            a, _ = memory._attention_gen_with_cached_text(q, k, v, state)
            b, _ = memory._attention_gen_with_cached_text(q, k, v, state, return_gen_lse=True)
        self.assertTrue(torch.equal(a["full_only_seq"], b["full_only_seq"]))
        _, expected = reference(
            q["full_only_seq"].unsqueeze(0),
            torch.cat((state.und_k_cached, k["full_only_seq"].unsqueeze(0)), 1),
            torch.cat((state.und_v_cached, v["full_only_seq"].unsqueeze(0)), 1),
            return_lse=True,
        )
        torch.testing.assert_close(b["_asi_gen_lse"], expected, rtol=0, atol=0)

    def test_bad_lse_shape_rejected(self):
        q, k, v = self.packs()
        with self.assertRaisesRegex(ValueError, "LSE shape"):
            fast.action_aligned_future_profiles(
                q["full_only_seq"],
                k["causal_seq"],
                k["full_only_seq"],
                v["causal_seq"],
                v["full_only_seq"],
                8**-0.5,
                (28, 3, 3),
                action_lse=torch.zeros(1, 4, 32),
            )


if __name__ == "__main__":
    unittest.main()
