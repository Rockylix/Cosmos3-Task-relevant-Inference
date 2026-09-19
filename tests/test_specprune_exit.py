import unittest

import torch

from cosmos_framework.inference.specprune_exit_plan import ExitHidden, ObservationPlan, top_mask
from cosmos_framework.inference.specprune_exit_metrics import future_vision


class ExitTest(unittest.TestCase):
    def test_future_axis_excludes_l0_not_channels(self):
        x = torch.arange(48 * 9 * 2 * 3).reshape(48, 9, 2, 3)
        self.assertTrue(torch.equal(future_vision(x), future_vision(x.unsqueeze(0))))
        self.assertEqual(future_vision(x.unsqueeze(0)).shape, (48, 8, 2, 3))
        self.assertTrue(torch.equal(future_vision(x)[:, 0], x[:, 1]))

    def test_future_axis_rejects_wrong_layout(self):
        with self.assertRaises(ValueError):
            future_vision(torch.zeros(2, 48, 9, 2, 3))
        with self.assertRaises(ValueError):
            future_vision(torch.zeros(9, 48, 2, 3))

    def test_restore_once_and_correct_values(self):
        x = torch.arange(60).reshape(10, 6).float()
        cache = ExitHidden(x)
        cache.put(torch.tensor([2, 5, 7]), x[[2, 5, 7]] + 10)
        rest = torch.tensor([0, 1, 3, 4, 6, 8, 9])
        cache.put(rest, x[rest] + 100)
        expected = x + 100
        expected[[2, 5, 7]] = x[[2, 5, 7]] + 10
        self.assertTrue(torch.equal(cache.finish(), expected))
        with self.assertRaises(AssertionError):
            cache.put(torch.tensor([2]), x[[2]])

    def test_missing_hidden_fails(self):
        cache = ExitHidden(torch.zeros(10, 6))
        cache.put(torch.tensor([0]), torch.ones(1, 6))
        with self.assertRaises(AssertionError):
            cache.finish()

    def test_cache_is_branch_local(self):
        x = torch.randn(10, 6)
        a, b = ExitHidden(x), ExitHidden(x)
        a.put(torch.arange(10), x)
        b.put(torch.arange(10), x + 1)
        self.assertTrue(torch.equal(a.finish(), x))
        self.assertTrue(torch.equal(b.finish(), x + 1))

    def test_full_keep_and_head_norm_once(self):
        x = torch.randn(10, 6)
        cache = ExitHidden(x)
        cache.put(torch.arange(10), x)
        layer = torch.nn.Sequential(torch.nn.RMSNorm(6), torch.nn.Linear(6, 4))
        self.assertTrue(torch.equal(layer(cache.finish()), layer(x)))

    def test_plan_nested_and_shared(self):
        plan = ObservationPlan()
        plan.begin(torch.ones(340, 12), 80)
        score = torch.arange(340).float()
        plan.local(0, score)
        plan.local(1, -score)
        plan.prune(10)
        self.assertEqual(plan.rows[-1]["scores_nonzero"], 0)
        for b in (14, 19, 24):
            plan.update(b, torch.arange(1, 341).float().expand(32, -1) / 1e5)
            plan.prune(b + 1)
        masks = list(plan.masks.values())
        for a, b in zip(masks, masks[1:]):
            self.assertFalse((b & ~a).any())
        self.assertGreaterEqual(int(masks[-1].sum()), 60)
        for mask in masks:
            future = mask.repeat(8).reshape(8, 340)
            self.assertTrue((future == future[:1]).all())

    def test_official_rank_confidence_ema(self):
        p = ObservationPlan()
        p.begin(torch.ones(340, 12), 80)
        attn = torch.rand(32, 340)
        avg = attn.mean(0)
        order = torch.argsort(avg, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ranks[order] = torch.arange(340)
        w = torch.sigmoid(-ranks.float())
        w /= w.sum() + 1e-8
        prob = (attn + 1e-8) / (attn + 1e-8).sum(-1, keepdim=True)
        h = -(prob * prob.log()).sum(-1).mean() / torch.tensor(340).log()
        expected = 0.2 * w / (h + 1e-8)
        p.update(14, attn)
        self.assertTrue(torch.allclose(p.importance, expected, rtol=1e-5))

    def test_observation_history_and_reset(self):
        p = ObservationPlan()
        rgb = torch.ones(340, 12)
        p.begin(rgb, 80)
        self.assertFalse(p.dynamic_mask.any())
        for b in p.config.global_blocks:
            p.global_scores[b] = torch.arange(340).float()
        p.finish()
        p.begin(rgb, 80)
        self.assertEqual(int(p.dynamic_mask.sum()), 27)
        p.reset()
        self.assertIsNone(p.previous_rgb)
        self.assertFalse(p.confidence)

    def test_top_mask_excludes_removed(self):
        score = torch.arange(10).float()
        allowed = score < 5
        self.assertEqual(torch.where(top_mask(score, 3, allowed))[0].tolist(), [2, 3, 4])

    def test_floor_stops_importance_and_confidence_updates(self):
        plan = ObservationPlan()
        plan.begin(torch.ones(340, 12), 80)
        plan.active[44:] = False
        plan.prune(10)
        self.assertFalse(plan.dynamic_enabled)
        plan.update(14, torch.ones(32, 340) / 1000)
        self.assertFalse(plan.confidence)
        self.assertFalse(plan.importance.any())
        self.assertIn(14, plan.action_scores)  # Diagnostic only, not selection.


if __name__ == "__main__":
    unittest.main()
