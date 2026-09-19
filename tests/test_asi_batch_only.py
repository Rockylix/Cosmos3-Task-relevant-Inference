import unittest
from unittest.mock import patch

import torch

from cosmos_framework.inference.asi_batch_only import BatchOnlyController, batch_only_profiles
from cosmos_framework.scripts import robolab_version1 as legacy


def reference_attention(*, query, key, value, scale, return_lse):
    h, kv = query.shape[2], key.shape[2]
    q = query.float().transpose(1, 2)
    k = key.float().repeat_interleave(h // kv, dim=2).transpose(1, 2)
    v = value.float().repeat_interleave(h // kv, dim=2).transpose(1, 2)
    z = (q * scale) @ k.transpose(-1, -2)
    return (z.softmax(-1) @ v).transpose(1, 2), z.logsumexp(-1).transpose(1, 2)


class BatchOnlyTest(unittest.TestCase):
    def fixture(self, spatial=7, heads=4, kv=2):
        torch.manual_seed(77)
        size = spatial * 9 + 33
        # Noncontiguous coordinates ensure B1 does not assume/cache geometry.
        order = torch.randperm(size)
        layout = dict(
            num_gen_tokens=size,
            latent_positions={f"L{f}": order[f * spatial : (f + 1) * spatial].tolist() for f in range(9)},
            action_queries=[
                dict(query_role="predicted", action_horizon=a, gen_position=int(order[9 * spatial + 1 + a]))
                for a in range(32)
            ],
        )
        return dict(
            torch=torch,
            q_gen=torch.randn(size, heads, 16),
            k_gen=torch.randn(size, kv, 16),
            k_ar=torch.randn(11, kv, 16),
            v_gen=torch.randn(size, kv, 16),
            v_ar=torch.randn(11, kv, 16),
            scaling=0.25,
            token_layout=layout,
        )

    def test_matches_legacy_multiple_gqa_layouts(self):
        for heads, kv in ((4, 2), (4, 4), (16, 8)):
            inputs = self.fixture(heads=heads, kv=kv)
            original = inputs["q_gen"].clone()
            with patch.object(legacy, "attention", side_effect=reference_attention) as kernel:
                a = legacy.action_aligned_future_profiles_with_lse(**inputs)
                b = batch_only_profiles(**inputs)
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-8)
            self.assertEqual(kernel.call_count, 2)  # One LSE call per scorer, unchanged.
            self.assertTrue(torch.equal(inputs["q_gen"], original))

    def test_controller_only_overrides_capture(self):
        self.assertEqual({k for k in BatchOnlyController.__dict__ if not k.startswith("__")}, {"_capture_profile"})
        self.assertIs(BatchOnlyController.end_stack, legacy.Version1Controller.end_stack)
        self.assertIs(BatchOnlyController._slice_pack_and_rope, legacy.Version1Controller._slice_pack_and_rope)

    def test_full_key_normalization(self):
        inputs = self.fixture()
        with patch.object(legacy, "attention", side_effect=reference_attention):
            scores = batch_only_profiles(**inputs)
        self.assertTrue(bool((scores.sum(-1) < 1).all()))

    def test_nonfinite_rejected(self):
        inputs = self.fixture()
        inputs["q_gen"].fill_(float("nan"))
        with patch.object(legacy, "attention", side_effect=reference_attention), self.assertRaises(RuntimeError):
            batch_only_profiles(**inputs)


if __name__ == "__main__":
    unittest.main()
