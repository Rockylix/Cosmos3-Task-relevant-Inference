import unittest

import torch

from cosmos_framework.inference.specprune_future import ChunkSelector, SpecPruneConfig, row_topk


class SelectorTest(unittest.TestCase):
    def test_dynamic_off_keeps_global_and_local_across_chunks(self):
        selector = ChunkSelector(SpecPruneConfig(dynamic_enabled=False))
        scores = torch.arange(340).float().repeat(8, 1)
        noise = torch.randn(8, 340, 64)
        selector.begin(noise)
        first, _ = selector.choose(scores, scores)
        for _ in range(3):
            history = scores.masked_fill(~selector.selected, torch.nan)
            selector.complete({13: history, 27: history})
            selector.begin(torch.randn_like(noise))
            mask, info = selector.choose(-scores, -scores)
            expected = row_topk(history, 40) | row_topk(-scores, 32)
            self.assertTrue(torch.equal(mask, expected))
            self.assertTrue(torch.all(mask.sum(-1) < 340))
            self.assertTrue((mask | ~first).all())
            self.assertEqual(info["dynamic_counts"], [0] * 8)
            self.assertIsNone(info["initial_noise_similarity_mean"])
            self.assertFalse(info["dynamic_enabled"])

    def test_first_chunk_local_union_and_single_selection(self):
        selector = ChunkSelector()
        noise = torch.randn(8, 340, 12)
        selector.begin(noise)
        scores = torch.arange(340).float().repeat(8, 1)
        mask, info = selector.choose(scores, -scores)
        self.assertTrue(torch.all(mask.sum(-1) == 64))
        self.assertTrue(info["first_chunk_local_only"])
        with self.assertRaises(RuntimeError):
            selector.choose(scores, scores)

    def test_dynamic_noise_can_protect_all_tokens(self):
        selector = ChunkSelector()
        torch.manual_seed(30)
        noise = torch.randn(8, 340, 64)
        scores = torch.rand(8, 340)
        selector.begin(noise)
        selector.choose(scores, scores)
        selector.complete({13: scores, 27: scores})
        selector.begin(torch.randn_like(noise))
        mask, info = selector.choose(scores, scores)
        self.assertTrue(mask.all())
        self.assertEqual(info["dynamic_counts"], [340] * 8)

    def test_stable_cap_and_missing_global(self):
        selector = ChunkSelector()
        noise = torch.randn(8, 340, 12)
        scores = torch.rand(8, 340)
        selector.begin(noise)
        mask, _ = selector.choose(scores, scores)
        history = scores.masked_fill(~mask, torch.nan)
        selector.complete({13: history, 27: history})
        selector.begin(noise)
        _, info = selector.choose(scores, scores)
        self.assertEqual(info["dynamic_counts"], [27] * 8)
        self.assertEqual(info["global_counts"], [32] * 8)
        selector.reset()
        self.assertIsNone(selector.previous_noise)

    def test_topk_does_not_fill_unobserved_slots(self):
        s = torch.full((8, 340), torch.nan)
        s[:, 3] = 0.5
        out = row_topk(s, 40)
        self.assertTrue(torch.all(out.sum(-1) == 1))
        self.assertTrue(out[:, 3].all())


if __name__ == "__main__":
    unittest.main()
